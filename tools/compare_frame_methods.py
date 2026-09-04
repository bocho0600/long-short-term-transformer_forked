# Compare long-memory frame-reduction methods at a matched frame budget:
#   * baseline              - all frames
#   * gate:uniform          - evenly-spaced subsample BEFORE the feature head (dumb)
#   * gate:norm             - cheap feature-norm gate BEFORE the feature head (Upgrade B)
#   * attention:select      - attention-guided selection AFTER the feature head (Upgrade A)
# Reports params + theoretical FLOPs + measured latency, so you can see which
# method reduces BOTH FLOPs and wall-clock (only pre-embedding pruning cuts the
# memory-bound feature head).
#
# Usage (GPU node):
#   python tools/compare_frame_methods.py \
#       --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_frameselect_singlepass.yaml \
#       --top_k 512 --device cuda --iters 100

import argparse
import time

import torch

import sys, os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'src'))

from rekognition_online_action_detection.config.defaults import get_cfg
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.models.feature_head import FEATURE_SIZES
from rekognition_online_action_detection.utils.flops import FlopCounter, count_parameters


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATA.NUM_CLASSES = args.num_classes
    cfg.DATA.CLASS_NAMES = [str(i) for i in range(args.num_classes)]
    cfg.DATA.IGNORE_INDEX = args.ignore_index
    cfg.DATA.METRICS = 'AP'
    cfg.DATA.FPS = args.fps
    L = cfg.MODEL.LSTR
    L.LONG_MEMORY_LENGTH = L.LONG_MEMORY_SECONDS * args.fps
    L.WORK_MEMORY_LENGTH = L.WORK_MEMORY_SECONDS * args.fps
    L.AGES_MEMORY_LENGTH = L.AGES_MEMORY_SECONDS * args.fps
    L.LONG_MEMORY_NUM_SAMPLES = L.LONG_MEMORY_LENGTH // L.LONG_MEMORY_SAMPLE_RATE
    L.WORK_MEMORY_NUM_SAMPLES = L.WORK_MEMORY_LENGTH // L.WORK_MEMORY_SAMPLE_RATE
    L.AGES_MEMORY_NUM_SAMPLES = L.AGES_MEMORY_LENGTH // L.AGES_MEMORY_SAMPLE_RATE
    return cfg


def make_inputs(cfg, device, batch_size):
    L = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    W = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
    vs = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
    ms = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
    visual = torch.randn(batch_size, L + W, vs, device=device)
    motion = torch.randn(batch_size, L + W, ms, device=device)
    mask = torch.zeros(batch_size, L, device=device)
    return visual, motion, mask, L


def apply_method(cfg, method, k):
    """Set the config flags for one method (all default OFF first)."""
    cfg.MODEL.LSTR.FRAME_SELECTION.ENABLED = False
    cfg.MODEL.LSTR.FRAME_GATE.ENABLED = False
    if method == 'gate:uniform':
        cfg.MODEL.LSTR.FRAME_GATE.ENABLED = True
        cfg.MODEL.LSTR.FRAME_GATE.SCORE = 'uniform'
        cfg.MODEL.LSTR.FRAME_GATE.TOP_K = k
    elif method == 'gate:norm':
        cfg.MODEL.LSTR.FRAME_GATE.ENABLED = True
        cfg.MODEL.LSTR.FRAME_GATE.SCORE = 'norm'
        cfg.MODEL.LSTR.FRAME_GATE.TOP_K = k
    elif method == 'gate:learned':
        cfg.MODEL.LSTR.FRAME_GATE.ENABLED = True
        cfg.MODEL.LSTR.FRAME_GATE.SCORE = 'learned'
        cfg.MODEL.LSTR.FRAME_GATE.TOP_K = k
    elif method == 'attention:select':
        cfg.MODEL.LSTR.FRAME_SELECTION.ENABLED = True
        cfg.MODEL.LSTR.FRAME_SELECTION.TOP_K = k


def measure(base_cfg, method, k, args, device):
    cfg = base_cfg.clone()
    apply_method(cfg, method, k)
    model = build_model(cfg, device)
    model.eval()
    params, _ = count_parameters(model)

    visual, motion, mask, L = make_inputs(cfg, device, args.batch_size)

    counter = FlopCounter()
    with torch.no_grad(), counter:
        model(visual, motion, mask)
    flops = counter.total_flops()

    is_cuda = device.type == 'cuda'
    with torch.no_grad():
        for _ in range(args.warmup):
            model(visual, motion, mask)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(visual, motion, mask)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / args.iters * 1e3
    return params, flops, latency_ms, L


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--ignore_index', type=int, default=21)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--top_k', type=int, default=512)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    base_cfg = build_cfg(args)
    k = args.top_k

    methods = ['baseline', 'gate:uniform', 'gate:norm', 'gate:learned', 'attention:select']
    rows = []
    for m in methods:
        params, flops, lat, L = measure(base_cfg, m, k, args, device)
        rows.append((m, params, flops, lat, L))

    L = rows[0][4]
    p0, f0, lat0 = rows[0][1], rows[0][2], rows[0][3]
    print('\nFrame-reduction methods @ TOP_K={} of {} frames '
          '(device={}, B={}, iters={})'.format(k, L, args.device, args.batch_size, args.iters))
    print('=' * 82)
    print('{:<20s} {:>12s} {:>12s} {:>10s} {:>12s} {:>10s}'.format(
        'method', 'params', 'GFLOPs', 'ΔFLOPs', 'latency ms', 'Δlat'))
    print('-' * 82)
    for (m, params, flops, lat, _) in rows:
        d_f = '' if m == 'baseline' else '{:+.1f}%'.format(100.0 * (flops - f0) / f0)
        d_l = '' if m == 'baseline' else '{:+.1f}%'.format(100.0 * (lat - lat0) / lat0)
        print('{:<20s} {:>12,d} {:>12.3f} {:>10s} {:>12.3f} {:>10s}'.format(
            m, params, flops / 1e9, d_f, lat, d_l))
    print('=' * 82)
    print('gate:* prune BEFORE the feature head (cut memory traffic -> should cut latency).')
    print('attention:select prunes AFTER it (cuts FLOPs only). params identical (no gate learns).')


if __name__ == '__main__':
    main()
