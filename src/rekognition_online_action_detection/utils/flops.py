# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight resource accounting: parameter counts and matmul FLOPs.

FLOPs are counted by monkeypatching F.linear and torch.bmm (which do ~all the
arithmetic in a transformer) and running one forward pass. FLOP = 2 x MACs.
Elementwise ops (softmax, layernorm, relu, dropout) are not counted.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F


class FlopCounter:
    """Counts MACs of F.linear and torch.bmm during a forward pass, optionally
    tagged per submodule via forward hooks."""

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
            self._add(self.stack[-1], rows * weight.shape[1] * weight.shape[0])
            return self._orig_linear(input, weight, bias)

        def bmm(a, b, *args, **kwargs):
            self._add(self.stack[-1],
                      a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2])
            return self._orig_bmm(a, b, *args, **kwargs)

        F.linear = linear
        torch.bmm = bmm
        return self

    def __exit__(self, *exc):
        F.linear = self._orig_linear
        torch.bmm = self._orig_bmm

    def hook(self, module, label):
        # Both hooks must return None: a forward hook that returns a value
        # replaces the module output, a pre-hook replaces the input.
        def pre(m, i):
            self.stack.append(label)

        def post(m, i, o):
            self.stack.pop()

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def total_flops(self):
        return 2 * sum(self.macs.values())


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def report_resources(cfg, model, sample_batch, device, logger,
                     num_train_windows=None, backward_factor=2.0):
    """Log parameter count, per-window forward FLOPs, and (optionally) an
    estimate of the whole training run's compute.

    Args:
        sample_batch: one batch from the train loader; everything but the last
            element is fed to the model (mirrors the trainer's forward call).
        num_train_windows: len(train dataset); if given, prints a training total.
    """
    total, trainable = count_parameters(model)
    logger.info('Model parameters: {:,} total | {:,} trainable | {:.1f} MB @ fp32'.format(
        total, trainable, total * 4 / 1e6))

    was_training = model.training
    model.eval()
    try:
        inputs = [x.to(device) for x in sample_batch[:-1]]
        batch_size = sample_batch[0].shape[0]
        counter = FlopCounter()
        with torch.no_grad(), counter:
            model(*inputs)
        fwd_per_window = counter.total_flops() / max(1, batch_size)
        logger.info('Forward compute: {:.3f} GFLOPs / window'.format(fwd_per_window / 1e9))

        if num_train_windows:
            epochs = cfg.SOLVER.NUM_EPOCHS
            train_step = fwd_per_window * (1.0 + backward_factor)  # fwd + bwd
            total_train = train_step * num_train_windows * epochs
            logger.info(
                'Estimated training compute: {:.3f} PFLOPs '
                '(fwd+bwd x {:,} windows x {} epochs)'.format(
                    total_train / 1e15, num_train_windows, epochs))
    finally:
        if was_training:
            model.train()
