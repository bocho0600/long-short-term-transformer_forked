# Smoke test for single-pass attention-guided long-memory frame selection
# (V2a, EViT-style key pruning). Run in the project env:
#   python tools/smoke_frame_selection.py
# Exercises the real edited transformer plumbing and the real LSTR helpers.
# No dataset/checkpoint needed.

import types
import torch

import sys, os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'src'))

from rekognition_online_action_detection.models import transformer as tr
from rekognition_online_action_detection.models.lstr import LSTR

torch.manual_seed(0)

D, H, Q, N, B, K = 64, 8, 16, 40, 3, 12

# Build a stage-1-like compressor: TransformerDecoder(1 layer) with LayerNorm.
layer = tr.TransformerDecoderLayer(D, H, 128, 0.0, 'relu')
stage1 = tr.TransformerDecoder(layer, 1, tr.layer_norm(D, True))
stage1.eval()

query = torch.randn(Q, B, D)
long_mem = torch.randn(N, B, D)
mask = torch.zeros(B, N)
mask[0, :5] = float('-inf')  # 5 padded frames for batch 0

# 1) Baseline (no selection) still works and weight exposure is unchanged.
out_base = stage1(query, long_mem, memory_key_padding_mask=mask)
out_w, weights = stage1(query, long_mem, memory_key_padding_mask=mask, need_weights=True)
assert torch.allclose(out_base, out_w, atol=1e-6), 'need_weights changed output!'
assert weights.shape == (B, H, Q, N)
print('[1] baseline + weight exposure: OK  (weights', tuple(weights.shape), ')')

# 2) Single-pass selection: correct shape, finite.
out_sp = stage1(query, long_mem, memory_key_padding_mask=mask, select_top_k=K)
assert out_sp.shape == (Q, B, D), tuple(out_sp.shape)
assert torch.isfinite(out_sp).all()
print('[2] single-pass output shape', tuple(out_sp.shape), '| finite: OK')

# 3) Single-pass == two-pass 'drop' (softmax over a subset == full softmax
#    restricted to that subset and renormalized).
dummy = types.SimpleNamespace(enc_modules=[stage1], frame_selection_top_k=K)
dummy._twopass = types.MethodType(LSTR._select_and_compress_long_memory, dummy)
out_tp = dummy._twopass(query, long_mem, mask)
assert torch.allclose(out_sp, out_tp, atol=1e-5), \
    'single-pass and two-pass disagree! max diff {}'.format((out_sp - out_tp).abs().max())
print('[3] single-pass == two-pass drop: OK')

# 4) k >= N is a no-op equal to baseline.
out_all = stage1(query, long_mem, memory_key_padding_mask=mask, select_top_k=N + 10)
assert torch.allclose(out_all, out_base, atol=1e-6), 'k>=N should equal baseline!'
print('[4] k>=N no-op equals baseline: OK')

# 5) Padded frames must never be selected.
_, w = stage1(query, long_mem, memory_key_padding_mask=mask, need_weights=True)
score = w.mean(1).mean(1) + mask
idx = torch.sort(score.topk(K, dim=1).indices, dim=1).values
assert not torch.isin(idx[0], torch.arange(5)).any(), 'a padded frame was selected!'
print('[5] padded frames excluded from top-k: OK')

print('\nALL SMOKE CHECKS PASSED')
