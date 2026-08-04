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
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        # Keep the un-reshaped (bsz, src_len) key padding mask for frame scoring.
        orig_key_padding_mask = key_padding_mask

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

        if select_top_k is not None:
            # Single-pass EViT-style frame selection: rank keys by the
            # attention they receive, keep the top-k, renormalize over them.
            attn_output, attn_output_weights = self._select_attention(
                q, k, v, mask, bsz, tsz, orig_key_padding_mask, select_top_k)
            need_weights = True
        elif need_weights:
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

    def _select_attention(self, q, k, v, mask, bsz, tsz,
                          key_padding_mask, top_k):
        """Single-pass top-k key selection (EViT-style).

        Computes the attention distribution over all keys once, ranks keys by
        the attention they receive (mean over heads and query tokens), keeps
        the top-k, and renormalizes the distribution over just those keys
        before the value aggregation. This yields the same result as recomputing
        the attention over only the selected keys (softmax over a subset equals
        the full softmax restricted to that subset and renormalized), but in a
        single pass -- no second compression pass.

        Args:
            q, k, v: (bsz * num_heads, seq, head_dim) projected/scaled tensors.
            mask: additive attention mask broadcast to (bsz * num_heads, tsz, src_len), or None.
            key_padding_mask: original (bsz, src_len) additive mask (0 / -inf), or None.
            top_k: number of keys to keep.
        Returns:
            attn_output: (bsz * num_heads, tsz, head_dim)
            weights: (bsz * num_heads, tsz, src_len) renormalized over kept keys.
        """
        src_len = k.shape[1]
        num_heads = self.num_heads

        weights = torch.bmm(q, k.transpose(1, 2))            # (B*H, tsz, src)
        if mask is not None:
            weights = weights + mask
        weights = F.softmax(weights, dim=-1)

        # Per-frame importance: mean over heads and query tokens -> (B, src).
        w4 = weights.view(bsz, num_heads, tsz, src_len)
        score = w4.mean(dim=1).mean(dim=1)                   # (B, src)
        # Never keep padded/masked frames.
        if key_padding_mask is not None:
            score = score + key_padding_mask

        keep_k = min(int(top_k), src_len)
        topk_idx = score.topk(keep_k, dim=1).indices         # (B, k)
        keep = torch.zeros(bsz, src_len, dtype=torch.bool, device=weights.device)
        keep.scatter_(1, topk_idx, True)                     # (B, src)

        # Zero out non-selected keys and renormalize over the kept ones.
        w4 = w4.masked_fill(~keep.view(bsz, 1, 1, src_len), 0.0)
        w4 = w4 / w4.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = w4.reshape(bsz * num_heads, tsz, src_len)

        weights = F.dropout(weights, p=self.dotproductattention.dropout,
                            training=self.training)
        attn_output = torch.bmm(weights, v)                  # (B*H, tsz, head_dim)
        return attn_output, weights


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
