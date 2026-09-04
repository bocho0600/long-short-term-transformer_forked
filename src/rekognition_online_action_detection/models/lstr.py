# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .feature_head import build_feature_head, FEATURE_SIZES


class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Attention-guided long-memory frame selection (inference-time only).
        self.frame_selection_enabled = cfg.MODEL.LSTR.FRAME_SELECTION.ENABLED
        self.frame_selection_top_k = cfg.MODEL.LSTR.FRAME_SELECTION.TOP_K
        self.frame_selection_mode = cfg.MODEL.LSTR.FRAME_SELECTION.MODE
        if self.frame_selection_enabled and self.frame_selection_mode != 'drop':
            raise NotImplementedError(
                "FRAME_SELECTION.MODE '{}' is not implemented yet; only 'drop' "
                'is supported.'.format(self.frame_selection_mode))
        # Log the selection status once on the first forward, so runs prove
        # whether pruning is actually active (and by how much).
        self._frame_selection_logged = False

        # Pre-embedding frame gate (Upgrade B): prune raw frames before the
        # feature head so the whole long pipeline processes fewer frames.
        self.frame_gate_enabled = cfg.MODEL.LSTR.FRAME_GATE.ENABLED
        self.frame_gate_top_k = cfg.MODEL.LSTR.FRAME_GATE.TOP_K
        self.frame_gate_score = cfg.MODEL.LSTR.FRAME_GATE.SCORE
        if self.frame_gate_enabled and self.frame_gate_score not in ('norm', 'uniform', 'learned'):
            raise ValueError("FRAME_GATE.SCORE must be 'norm', 'uniform' or "
                             "'learned', got {}".format(self.frame_gate_score))
        if self.frame_gate_enabled and self.frame_gate_score == 'learned':
            # Tiny learned scorer over RAW (visual+motion) features -> 1 score/frame.
            with_visual = 'motion' not in cfg.INPUT.MODALITY
            with_motion = 'visual' not in cfg.INPUT.MODALITY
            gate_in = 0
            if with_visual:
                gate_in += FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            if with_motion:
                gate_in += FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            self.frame_gate_scorer = nn.Linear(gate_in, 1)
        self._frame_gate_logged = False

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for param in cfg.MODEL.LSTR.ENC_MODULE:
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerDecoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
        else:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerEncoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = tr.TransformerEncoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def _select_and_compress_long_memory(self, query, long_memories,
                                         memory_key_padding_mask):
        """Attention-guided top-k long-memory frame selection before stage-1
        compression (Variant 2 / V2a, 'drop' mode, inference-time only).

        A first stage-1 pass scores each frame by the cross-attention weight it
        receives (mean over heads and over the query tokens); the top-k frames
        are then re-compressed on their own, so the 16 compressed tokens
        summarize only the selected frames. Downstream shapes are unchanged.

        Args:
            query: (Q, B, D) stage-1 learnable queries (Q = 16).
            long_memories: (N, B, D) positional-encoded long-memory frames.
            memory_key_padding_mask: (B, N) additive float mask (0 / -inf), or None.
        Returns:
            (Q, B, D) compressed long-memory tokens.
        """
        num_frames = long_memories.shape[0]
        top_k = self.frame_selection_top_k

        # Nothing to prune -> identical to the baseline stage-1 compression.
        if top_k >= num_frames:
            return self.enc_modules[0](
                query, long_memories,
                memory_key_padding_mask=memory_key_padding_mask)

        # Pass 1: score frames from the stage-1 cross-attention weights.
        _, attn_weights = self.enc_modules[0](
            query, long_memories,
            memory_key_padding_mask=memory_key_padding_mask,
            need_weights=True)                       # (B, H, Q, N)
        score = attn_weights.mean(dim=1).mean(dim=1)  # (B, N)

        # Never select masked (padded/oracle) frames: -inf sinks them below top-k.
        if memory_key_padding_mask is not None:
            score = score + memory_key_padding_mask

        topk_idx = score.topk(top_k, dim=1).indices   # (B, k)
        topk_idx, _ = torch.sort(topk_idx, dim=1)      # keep temporal order

        # Gather the selected frames: (N, B, D) -> (k, B, D).
        idx = topk_idx.transpose(0, 1).unsqueeze(-1).expand(
            top_k, -1, long_memories.shape[-1])
        selected = torch.gather(long_memories, 0, idx)

        # Gather the matching mask entries: (B, N) -> (B, k).
        if memory_key_padding_mask is not None:
            selected_mask = torch.gather(memory_key_padding_mask, 1, topk_idx)
        else:
            selected_mask = None

        # Pass 2: compress over the selected frames only.
        return self.enc_modules[0](
            query, selected, memory_key_padding_mask=selected_mask)

    def _apply_frame_gate(self, long_visual, long_motion, memory_key_padding_mask):
        """Pre-embedding frame gate (Upgrade B).

        Scores the RAW long-memory frames with a cheap signal, keeps the top-k,
        and runs the feature head + positional encoding on ONLY those k frames.
        Because fewer frames flow through the whole long pipeline, this reduces
        FLOPs *and* activation-memory traffic (unlike attention selection, which
        runs after the feature head).

        Args:
            long_visual: (B, N, vis) raw visual features.
            long_motion: (B, N, mot) raw motion features.
            memory_key_padding_mask: (B, N) additive mask (0 / -inf), or None.
        Returns:
            (long_memories, gated_mask): (k, B, d) positional-encoded embeddings
            and the (B, k) mask for the kept frames (or None).
        """
        B, N = long_visual.shape[0], long_visual.shape[1]
        k = min(int(self.frame_gate_top_k), N)

        if not self._frame_gate_logged:
            if k >= N:
                print('[LSTR] frame gate ENABLED but TOP_K ({}) >= N ({}) -> '
                      'keeping ALL frames (no-op)'.format(self.frame_gate_top_k, N),
                      flush=True)
            else:
                print('[LSTR] pre-embedding frame gate ACTIVE: keeping {} / {} raw '
                      'frames (score={})'.format(k, N, self.frame_gate_score), flush=True)
            self._frame_gate_logged = True

        raw_score = None
        if self.frame_gate_score == 'uniform':
            idx = torch.linspace(0, N - 1, k, device=long_visual.device).round().long()
            idx = idx.unsqueeze(0).expand(B, k)
        elif self.frame_gate_score == 'learned':
            # Tiny learned scorer on the RAW features -> one score per frame.
            raw = torch.cat((long_visual, long_motion), dim=-1)           # (B, N, gate_in)
            raw_score = self.frame_gate_scorer(raw).squeeze(-1)           # (B, N)
            score = raw_score
            if memory_key_padding_mask is not None:
                score = score + memory_key_padding_mask
            idx = score.topk(k, dim=1).indices                            # (B, k)
        else:  # 'norm' -- cheap saliency from raw feature magnitude
            score = long_visual.norm(dim=-1) + long_motion.norm(dim=-1)   # (B, N)
            if memory_key_padding_mask is not None:
                score = score + memory_key_padding_mask                   # sink padded frames
            idx = score.topk(k, dim=1).indices                            # (B, k)
        idx, _ = torch.sort(idx, dim=1)                                   # keep temporal order

        # Gather the raw frames for the kept indices.
        sel_visual = torch.gather(
            long_visual, 1, idx.unsqueeze(-1).expand(B, k, long_visual.shape[-1]))
        sel_motion = torch.gather(
            long_motion, 1, idx.unsqueeze(-1).expand(B, k, long_motion.shape[-1]))

        # Embed only the kept frames.
        emb = self.feature_head_long(sel_visual, sel_motion).transpose(0, 1)   # (k, B, d)

        # Learned gate: hard top-k is non-differentiable, so multiply the kept
        # embeddings by a soft sigmoid weight -> gradients reach the scorer,
        # teaching it to score useful frames higher.
        if raw_score is not None:
            gate_w = torch.sigmoid(torch.gather(raw_score, 1, idx))       # (B, k)
            emb = emb * gate_w.transpose(0, 1).unsqueeze(-1)              # (k, B, 1)

        # Positional encoding at the ORIGINAL frame positions (not 0..k-1).
        pe = self.pos_encoding.pe.squeeze(1)[idx].transpose(0, 1)              # (k, B, d)
        long_memories = self.pos_encoding.dropout(emb + pe)

        gated_mask = (torch.gather(memory_key_padding_mask, 1, idx)
                      if memory_key_padding_mask is not None else None)
        return long_memories, gated_mask

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            long_visual = visual_inputs[:, :self.long_memory_num_samples]
            long_motion = motion_inputs[:, :self.long_memory_num_samples]
            if self.frame_gate_enabled:
                # Upgrade B: prune raw frames BEFORE the feature head, so the
                # feature head + stage-1 process only the kept frames.
                long_memories, memory_key_padding_mask = self._apply_frame_gate(
                    long_visual, long_motion, memory_key_padding_mask)
            else:
                # Compute long memories
                long_memories = self.pos_encoding(self.feature_head_long(
                    long_visual, long_motion).transpose(0, 1))

            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    if self.frame_selection_enabled:
                        if not self._frame_selection_logged:
                            n_frames = long_memories.shape[0]
                            keep = min(self.frame_selection_top_k, n_frames)
                            if keep >= n_frames:
                                print('[LSTR] frame selection ENABLED but TOP_K '
                                      '({}) >= N ({}) -> keeping ALL frames '
                                      '(no-op)'.format(self.frame_selection_top_k,
                                                       n_frames), flush=True)
                            else:
                                print('[LSTR] single-pass frame selection ACTIVE: '
                                      'keeping {} / {} long-memory frames '
                                      '(mode={})'.format(keep, n_frames,
                                                         self.frame_selection_mode),
                                      flush=True)
                            self._frame_selection_logged = True
                        # Single-pass: prune keys inside the stage-1 attention.
                        long_memories = self.enc_modules[0](
                            enc_queries[0], long_memories,
                            memory_key_padding_mask=memory_key_padding_mask,
                            select_top_k=self.frame_selection_top_k)
                    else:
                        long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                            memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                visual_inputs[:, self.long_memory_num_samples:],
                motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)


@registry.register('LSTR')
class LSTRStream(LSTR):

    def __init__(self, cfg):
        super(LSTRStream, self).__init__(cfg)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None

    def stream_inference(self,
                         long_visual_inputs,
                         long_motion_inputs,
                         work_visual_inputs,
                         work_motion_inputs,
                         memory_key_padding_mask=None):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)

            if self.long_memories_cache is None:
                self.long_memories_cache = long_memories
            else:
                self.long_memories_cache = torch.cat((
                    self.long_memories_cache[1:], long_memories
                ))

            long_memories = self.long_memories_cache
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask)
            self.compressed_long_memories_cache  = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)
