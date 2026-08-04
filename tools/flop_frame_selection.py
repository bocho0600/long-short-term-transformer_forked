# FLOP measurement for LSTR long-memory frame selection.
#
# Builds the REAL model and counts matmul FLOPs (nn.Linear / F.linear and
# torch.bmm -- the attention and projection ops) for a single inference window,
# with a per-stage breakdown, comparing the baseline against frame selection at
# several TOP_K values. No dataset or checkpoint required.
#
# Usage (in the project env):
#   python tools/flop_frame_selection.py \
#       --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_frameselect_singlepass.yaml \
#       --top_k 64 128 256
#
# Notes:
#   * FLOP = 2 x MACs. softmax / layernorm / relu / pos-encoding are elementwise
#     and not counted (negligible next to the matmuls).
#   * FPS defaults to 1 (THUMOS in this repo: LONG 2048s / rate 4 -> 512 frames,
#     WORK 8s -> 8 frames). Pass --fps to match your data_info if different; the
#     script prints the derived frame counts so you can sanity-check them.

import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F

import sys, os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'src'))

from rekognition_online_action_detection.config.defaults import get_cfg
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.models.feature_head import FEATURE_SIZES


class FlopCounter:
    """Monkeypatches F.linear and torch.bmm to accumulate MACs, tagged by a
    label stack that module hooks push/pop for a per-stage breakdown."""

    def __init__(self):
        self.macs = OrderedDict()
        self.stack = ['other']
        self._orig_linear = None
        self._orig_bmm = None

    def _add(self, label, macs):
        self.macs[label] = self.macs.get(label, 0) + int(macs)

    def __enter__(self):
        self._orig_linear = F.linear
        self._orig_bmm = torch.bmm

        def linear(input, weight, bias=None):
            rows = input.numel() // input.shape[-1]
            in_f = weight.shape[1]
            out_f = weight.shape[0]
            self._add(self.stack[-1], rows * in_f * out_f)
            return self._orig_linear(input, weight, bias)

        def bmm(a, b, *args, **kwargs):
            # a: (B, n, m), b: (B, m, p) -> B*n*m*p MACs
            self._add(self.stack[-1], a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2])
            return self._orig_bmm(a, b, *args, **kwargs)

        F.linear = linear
        torch.bmm = bmm
        return self

    def __exit__(self, *exc):
        F.linear = self._orig_linear
        torch.bmm = self._orig_bmm

    def hook(self, module, label):
        module.register_forward_pre_hook(lambda m, i: self.stack.append(label))
        module.register_forward_hook(lambda m, i, o: self.stack.pop())

    def total_flops(self):
        return 2 * sum(self.macs.values())


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Fields normally filled from data_info -- set manually so no dataset is needed.
    cfg.DATA.NUM_CLASSES = args.num_classes
    cfg.DATA.CLASS_NAMES = [str(i) for i in range(args.num_classes)]
    cfg.DATA.IGNORE_INDEX = args.ignore_index
    cfg.DATA.METRICS = 'AP'
    cfg.DATA.FPS = args.fps
    # Derived memory sizes (mirrors assert_and_infer_cfg).
    L = cfg.MODEL.LSTR
    L.LONG_MEMORY_LENGTH = L.LONG_MEMORY_SECONDS * args.fps
    L.WORK_MEMORY_LENGTH = L.WORK_MEMORY_SECONDS * args.fps
    L.AGES_MEMORY_LENGTH = L.AGES_MEMORY_SECONDS * args.fps
    L.LONG_MEMORY_NUM_SAMPLES = L.LONG_MEMORY_LENGTH // L.LONG_MEMORY_SAMPLE_RATE
    L.WORK_MEMORY_NUM_SAMPLES = L.WORK_MEMORY_LENGTH // L.WORK_MEMORY_SAMPLE_RATE
    L.AGES_MEMORY_NUM_SAMPLES = L.AGES_MEMORY_LENGTH // L.AGES_MEMORY_SAMPLE_RATE
    return cfg


def measure(cfg, enabled, top_k, args):
    cfg = cfg.clone()
    cfg.MODEL.LSTR.FRAME_SELECTION.ENABLED = enabled
    if top_k is not None:
        cfg.MODEL.LSTR.FRAME_SELECTION.TOP_K = top_k

    model = build_model(cfg, torch.device('cpu'))
    model.eval()

    L = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    W = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
    visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
    motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
    B = args.batch_size

    visual = torch.randn(B, L + W, visual_size)
    motion = torch.randn(B, L + W, motion_size)
    mask = torch.zeros(B, L)

    counter = FlopCounter()
    counter.hook(model.feature_head_long, 'long_feature_head')
    counter.hook(model.feature_head_work, 'work_feature_head')
    counter.hook(model.enc_modules[0], 'stage1_encoder')
    if len(model.enc_modules) > 1:
        counter.hook(model.enc_modules[1], 'stage2_encoder')
    counter.hook(model.dec_modules, 'decoder')
    counter.hook(model.classifier, 'classifier')

    with torch.no_grad(), counter:
        model(visual, motion, mask)
    return counter, L, W


def fmt(flops):
    return '{:8.3f} GFLOPs'.format(flops / 1e9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True)
    p.add_argument('--fps', type=int, default=1)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--ignore_index', type=int, default=21)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--top_k', type=int, nargs='*', default=[64, 128, 256])
    # Optional: estimate total FLOPs for a whole training run.
    p.add_argument('--epochs', type=int, default=None,
                   help='if set, also print an estimated total training-run FLOPs')
    p.add_argument('--train_iters', type=int, default=None,
                   help='training iterations per epoch (tqdm count of the train phase)')
    p.add_argument('--test_iters', type=int, default=None,
                   help='test iterations per epoch (tqdm count of the test phase)')
    p.add_argument('--backward_factor', type=float, default=2.0,
                   help='backward FLOPs as a multiple of forward (default 2.0)')
    args = p.parse_args()

    base_cfg = build_cfg(args)

    # Baseline (selection off).
    base_counter, L, W = measure(base_cfg, enabled=False, top_k=None, args=args)
    base_total = base_counter.total_flops()

    print('\nLSTR FLOPs per inference window '
          '(B={}, long={} frames, work={} frames, d_model={})'.format(
              args.batch_size, L, W,
              base_cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES))
    print('=' * 66)
    print('\n[baseline]  (FRAME_SELECTION.ENABLED = False)')
    for k, v in base_counter.macs.items():
        print('  {:<20s} {}'.format(k, fmt(2 * v)))
    print('  {:<20s} {}'.format('TOTAL', fmt(base_total)))

    for top_k in args.top_k:
        counter, _, _ = measure(base_cfg, enabled=True, top_k=top_k, args=args)
        total = counter.total_flops()
        print('\n[frame-selection]  TOP_K = {} / {}'.format(top_k, L))
        for kk, v in counter.macs.items():
            print('  {:<20s} {}'.format(kk, fmt(2 * v)))
        print('  {:<20s} {}   ({:+.1f}% vs baseline)'.format(
            'TOTAL', fmt(total), 100.0 * (total - base_total) / base_total))

    # Optional whole-training-run estimate (analytic; not measured live).
    if args.epochs is not None:
        fwd = base_total  # per-window forward FLOPs (batch training uses baseline path)
        train_step = fwd * (1.0 + args.backward_factor)  # forward + backward
        print('\n' + '=' * 66)
        print('Training-run FLOP estimate (per window, B={})'.format(args.batch_size))
        print('  forward / window      {}'.format(fmt(fwd)))
        print('  backward / window     {}   (x{} of forward)'.format(
            fmt(fwd * args.backward_factor), args.backward_factor))
        print('  train step (fwd+bwd)  {}'.format(fmt(train_step)))
        if args.train_iters is not None:
            total_train = train_step * args.train_iters * args.epochs
            print('  -> train phase total  {}   ({} iters x {} epochs)'.format(
                fmt(total_train), args.train_iters, args.epochs))
            if args.test_iters is not None:
                total_test = fwd * args.test_iters * args.epochs  # test = forward only
                print('  -> test  phase total  {}   ({} iters x {} epochs)'.format(
                    fmt(total_test), args.test_iters, args.epochs))
                grand = total_train + total_test
                print('  -> WHOLE RUN total    {}  ({:.2f} PFLOPs)'.format(
                    fmt(grand), grand / 1e15))
        else:
            print('  (pass --train_iters [and --test_iters] to get run totals)')


if __name__ == '__main__':
    main()
