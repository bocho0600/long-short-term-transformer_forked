# Benchmark: MEASURED wall-clock latency + parameter count + FLOPs for the
# baseline vs frame selection at several TOP_K. Complements
# flop_frame_selection.py (theoretical FLOPs) with real timing, so you can show
# whether the ~FLOP saving actually translates into a speedup on hardware.
#
# Usage (best on the GPU node):
#   python tools/benchmark_frame_selection.py \
#       --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_frameselect_singlepass.yaml \
#       --top_k 64 128 256 --device cuda --iters 100
#
# Notes:
#   * Parameters are IDENTICAL for every setting (selection adds none) -- reported
#     to prove zero parameter overhead.
#   * FLOPs are theoretical (matmul count). Wall-clock is measured and may differ
#     from the FLOP ratio (memory-bound feature head, gather overhead, etc.).
#   * Run on the same device you care about; CPU timings are not representative
#     of the A100.

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
    return visual, motion, mask, L, W


def bench_one(base_cfg, enabled, top_k, args, device):
    cfg = base_cfg.clone()
    cfg.MODEL.LSTR.FRAME_SELECTION.ENABLED = enabled
    if top_k is not None:
        cfg.MODEL.LSTR.FRAME_SELECTION.TOP_K = top_k

    model = build_model(cfg, device)
    model.eval()
    params, _ = count_parameters(model)

    visual, motion, mask, L, W = make_inputs(cfg, device, args.batch_size)

    # Theoretical FLOPs (one forward).
    counter = FlopCounter()
    with torch.no_grad(), counter:
        model(visual, motion, mask)
    flops = counter.total_flops()

    # Measured latency.
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
    return params, flops, latency_ms, L, W


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--ignore_index', type=int, default=21)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--top_k', type=int, nargs='*', default=[64, 128, 256])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    base_cfg = build_cfg(args)

    p0, f0, lat0, L, W = bench_one(base_cfg, enabled=False, top_k=None, args=args, device=device)

    print('\nBenchmark: LSTR baseline vs frame selection '
          '(device={}, B={}, long={} frames, iters={})'.format(
              args.device, args.batch_size, L, args.iters))
    print('=' * 78)
    hdr = '{:<24s} {:>12s} {:>14s} {:>16s}'.format('setting', 'params', 'GFLOPs/win', 'latency ms/win')
    print(hdr)
    print('-' * 78)
    print('{:<24s} {:>12,d} {:>14.3f} {:>16.3f}'.format(
        'baseline (OFF)', p0, f0 / 1e9, lat0))

    for k in args.top_k:
        params, flops, lat, _, _ = bench_one(base_cfg, enabled=True, top_k=k, args=args, device=device)
        d_flops = 100.0 * (flops - f0) / f0
        d_lat = 100.0 * (lat - lat0) / lat0
        d_params = params - p0
        print('{:<24s} {:>12,d} {:>14.3f} {:>16.3f}'.format(
            'select TOP_K={}'.format(k), params, flops / 1e9, lat))
        print('{:<24s} {:>12s} {:>13.1f}% {:>15.1f}%'.format(
            '  vs baseline', ('+{}'.format(d_params) if d_params else 'same'),
            d_flops, d_lat))

    print('=' * 78)
    print('params: identical across settings (selection adds no parameters).')
    print('GFLOPs: theoretical matmul count. latency: measured wall-clock '
          '(may differ from the FLOP ratio).')


if __name__ == '__main__':
    main()
