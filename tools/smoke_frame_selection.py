# Smoke test for attention-guided long-memory frame selection (V2a, 'drop').
# Run in the project env:  python tools/smoke_frame_selection.py
# Exercises the real edited transformer plumbing and the real LSTR selection
# helper. No dataset/checkpoint needed.

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

# 1) Exposing weights must NOT change the compressed output.
out_plain = stage1(query, long_mem, memory_key_padding_mask=mask)
out_w, weights = stage1(query, long_mem, memory_key_padding_mask=mask, need_weights=True)
assert torch.allclose(out_plain, out_w, atol=1e-6), 'need_weights changed the output!'
assert weights.shape == (B, H, Q, N), 'weights shape {}'.format(tuple(weights.shape))
print('[1] weights shape', tuple(weights.shape), '| output unchanged: OK')

# 2) Real selection helper via a lightweight stand-in bound to the real method.
dummy = types.SimpleNamespace(
    enc_modules=[stage1],
    frame_selection_top_k=K,
)
dummy._select = types.MethodType(LSTR._select_and_compress_long_memory, dummy)
compressed = dummy._select(query, long_mem, mask)
assert compressed.shape == (Q, B, D), 'compressed shape {}'.format(tuple(compressed.shape))
assert torch.isfinite(compressed).all(), 'non-finite output'
print('[2] selection output shape', tuple(compressed.shape), '| finite: OK')

# 3) k >= N is a no-op (identical to baseline compression).
dummy.frame_selection_top_k = N + 10
noop = dummy._select(query, long_mem, mask)
assert torch.allclose(noop, out_plain, atol=1e-6), 'k>=N should equal baseline!'
print('[3] k>=N no-op equals baseline: OK')

# 4) Padded frames must never be selected. Re-derive the top-k the helper uses.
_, w = stage1(query, long_mem, memory_key_padding_mask=mask, need_weights=True)
score = w.mean(1).mean(1) + mask
idx = torch.sort(score.topk(K, dim=1).indices, dim=1).values
assert not torch.isin(idx[0], torch.arange(5)).any(), 'a padded frame was selected!'
print('[4] padded frames excluded from top-k: OK')

print('\nALL SMOKE CHECKS PASSED')
