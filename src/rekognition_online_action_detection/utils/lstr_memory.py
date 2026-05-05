# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_right

import numpy as np


def resolve_long_memory_window(work_start,
                               long_memory_length,
                               window_length=0,
                               window_stride=0,
                               window_index=0):
    full_start = max(0, work_start - long_memory_length)
    full_end = work_start - 1
    if window_length <= 0:
        return full_start, full_end

    stride = window_stride if window_stride > 0 else window_length
    window_end = full_end - window_index * stride
    if window_end < full_start:
        return 0, -1

    window_start = max(full_start, window_end - window_length + 1)
    return window_start, window_end


def segment_sampler(start, end, num_samples):
    indices = np.linspace(start, end, num_samples)
    return np.sort(indices).astype(np.int32)


def uniform_sampler(start, end, num_samples, sample_rate):
    indices = np.arange(start, end + 1)[::sample_rate]
    padding = num_samples - indices.shape[0]
    if padding > 0:
        indices = np.concatenate((np.zeros(padding), indices))
    return np.sort(indices).astype(np.int32)


def memory_key_padding_mask_from_indices(indices):
    memory_key_padding_mask = np.zeros(indices.shape[0])
    last_zero = bisect_right(indices, 0) - 1
    if last_zero > 0:
        memory_key_padding_mask[:last_zero] = float('-inf')
    return memory_key_padding_mask
