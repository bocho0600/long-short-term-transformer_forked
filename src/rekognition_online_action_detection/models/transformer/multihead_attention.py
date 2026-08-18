# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout

    def forward(self, q, k, v, attn_mask=None, need_weights=False):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)
        if need_weights:
            return attn_output, attn_output_weights
        return attn_output


class DotProductAttentionStream(DotProductAttention):

    def __init__(self, dropout=0.0):
        super(DotProductAttentionStream, self).__init__(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.k_weights_cache = None
        self.k_pos_weights_cache = None

    def stream_inference(self, q, k, v, k_pos, v_pos, attn_mask=None):
        if self.k_weights_cache is not None:
            k_weights_new = torch.bmm(q, k[:, [-1]].transpose(1, 2))
            k_weights = torch.cat((self.k_weights_cache[:, :, 1:], k_weights_new), dim=-1)
            self.k_weights_cache = k_weights
            k_pos_weights = self.k_pos_weights_cache
        else:
            k_weights = torch.bmm(q, k.transpose(1, 2))
            self.k_weights_cache = k_weights
            k_pos_weights = torch.bmm(q, k_pos.transpose(1, 2))
            self.k_pos_weights_cache = k_pos_weights
        attn_output_weights = k_weights + k_pos_weights

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, (v + v_pos))
        return attn_output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        self.dotproductattention = DotProductAttention(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None,
                need_weights=False, select_top_k=None):
        if select_top_k is not None:
            # Single-pass EViT-style frame selection with GATHER + deferred
            # V-projection: score keys, keep the top-k, then value-project and
            # aggregate ONLY those k frames, so the attention cost shrinks with k.
            return self._select_forward_gather(
                q, k, v, attn_mask, key_padding_mask, select_top_k)

        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if need_weights:
            attn_output, attn_output_weights = self.dotproductattention(
                q, k, v, mask, need_weights=True)
        else:
            attn_output = self.dotproductattention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        if need_weights:
            # (bsz * num_heads, tsz, src_len) -> (bsz, num_heads, tsz, src_len).
            # The flattened batch dim is batch-major, head-minor (b * H + h).
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tsz, -1)
            return self.out_proj(attn_output), attn_output_weights
        return self.out_proj(attn_output), None

    def _select_forward_gather(self, q_in, k_in, v_in, attn_mask,
                               key_padding_mask, top_k):
        """Single-pass top-k frame selection with gather + deferred V-projection.

        Q and all K are projected (K is needed to score every frame). The top-k
        frames are then selected, and ONLY those k frames are value-projected and
        aggregated -- so the V-projection and the value matmul cost scale with k,
        not the full sequence length. The K-projection and the score matmul stay
        full (attention needs every key to rank it), so this reduces compute but
        cannot touch the feature head or the K-projection.

        The output is mathematically identical to computing softmax over only the
        selected keys (== the masking form, renormalized), just cheaper.

        Args:
            q_in: (tsz, bsz, embed) query input.
            k_in, v_in: (src, bsz, embed) memory input (same tensor for cross-attn).
            attn_mask: additive (tsz, src) mask or None.
            key_padding_mask: additive (bsz, src) mask (0 / -inf) or None.
            top_k: number of frames to keep.
        Returns:
            (out_proj(attn_output), None): (tsz, bsz, embed) output.
        """
        tsz, bsz, embed_dim = q_in.shape
        src_len = k_in.shape[0]
        H = self.num_heads
        head_dim = embed_dim // H
        scaling = float(head_dim) ** -0.5
        keep_k = min(int(top_k), src_len)

        b = self.in_proj_bias
        # Project Q and all K (K needed to score every frame).
        wq, bq = self.in_proj_weight[:embed_dim], (b[:embed_dim] if b is not None else None)
        wk, bk = self.in_proj_weight[embed_dim:2 * embed_dim], (b[embed_dim:2 * embed_dim] if b is not None else None)
        q = F.linear(q_in, wq, bq) * scaling
        k = F.linear(k_in, wk, bk)
        q = q.contiguous().view(tsz, bsz * H, head_dim).transpose(0, 1)       # (B*H, tsz, hd)
        k = k.contiguous().view(src_len, bsz * H, head_dim).transpose(0, 1)   # (B*H, src, hd)

        scores = torch.bmm(q, k.transpose(1, 2))                             # (B*H, tsz, src)

        # Additive mask over all keys (for scoring).
        mask = None
        if attn_mask is not None:
            am = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            am = am.unsqueeze(1).repeat(1, H, 1, 1).reshape(-1, tsz, src_len)
            mask = am
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            kpm = kpm.unsqueeze(1).repeat(1, H, 1, 1).reshape(-1, tsz, src_len)
            mask = kpm if mask is None else mask + kpm
        if mask is not None:
            scores = scores + mask

        # Per-frame importance from the full attention distribution.
        probs = F.softmax(scores, dim=-1)
        fscore = probs.view(bsz, H, tsz, src_len).mean(dim=1).mean(dim=1)     # (B, src)
        if key_padding_mask is not None:
            fscore = fscore + key_padding_mask                               # sink padded frames

        topk_idx = fscore.topk(keep_k, dim=1).indices                        # (B, k)

        # Deferred V-projection: gather RAW v for the selected frames, project only those.
        v_gather = topk_idx.transpose(0, 1).unsqueeze(-1).expand(keep_k, bsz, embed_dim)
        v_sel_in = torch.gather(v_in, 0, v_gather)                           # (k, bsz, embed)
        wv, bv = self.in_proj_weight[2 * embed_dim:], (b[2 * embed_dim:] if b is not None else None)
        v_sel = F.linear(v_sel_in, wv, bv)
        v_sel = v_sel.contiguous().view(keep_k, bsz * H, head_dim).transpose(0, 1)  # (B*H, k, hd)

        # Gather the score columns for the selected keys; softmax over just those.
        idx_bh = topk_idx.unsqueeze(1).expand(bsz, H, keep_k).reshape(bsz * H, keep_k)
        idx_cols = idx_bh.unsqueeze(1).expand(bsz * H, tsz, keep_k)
        scores_sel = torch.gather(scores, 2, idx_cols)                       # (B*H, tsz, k)
        weights = F.softmax(scores_sel, dim=-1)
        weights = F.dropout(weights, p=self.dotproductattention.dropout,
                            training=self.training)

        attn_output = torch.bmm(weights, v_sel)                             # (B*H, tsz, hd)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz, embed_dim)
        return self.out_proj(attn_output), None


class MultiheadAttentionStream(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        super(MultiheadAttentionStream, self).__init__(embed_dim, num_heads, dropout, bias, kdim, vdim)

        self.dotproductattention = DotProductAttentionStream(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.k_pos_cache = None
        self.v_pos_cache = None

    def stream_inference(self, q, k, v, pos, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if self.q_cache is not None:
            q = self.q_cache
        else:
            _b = self.in_proj_bias
            _start = None
            _end = embed_dim
            _w = self.in_proj_weight[:_end, :]
            if _b is not None:
                _b = _b[:_end]
            q = F.linear(q, _w, _b)
            self.q_cache = q

        assert (self.k_cache is None) == (self.k_pos_cache is None)
        if self.k_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_new = F.linear(k[[-1]], _w, None)
            k = torch.cat((self.k_cache[1:], k_new))
            self.k_cache = k
            k_pos = self.k_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(k, _w, None)
            self.k_cache = k
            k_pos = F.linear(pos, _w, _b)
            self.k_pos_cache = k_pos

        assert (self.v_cache is None) == (self.v_pos_cache is None)
        if self.v_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_new = F.linear(v[[-1]], _w, None)
            v = torch.cat((self.v_cache[1:], v_new))
            self.v_cache = v
            v_pos = self.v_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, None)
            self.v_cache = v
            v_pos = F.linear(pos, _w, _b)
            self.v_pos_cache = v_pos

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k_pos = k_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v_pos = v_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.dotproductattention.stream_inference(q, k, v, k_pos, v_pos, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None
