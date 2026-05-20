# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np

from .datasets import DATA_LAYERS as registry


@registry.register('LSTRTHUMOS')
@registry.register('LSTRTVSeries')
class LSTRDataLayer(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.long_memory_mask_frame = cfg.MODEL.LSTR.LONG_MEMORY_MASK_FRAME
        self.long_memory_mask_ratio = cfg.MODEL.LSTR.LONG_MEMORY_MASK_RATIO
        self.long_memory_masked_frames_bias = cfg.MODEL.LSTR.LONG_MEMORY_MASKED_FRAMES_BIAS
        self.long_memory_unmasked_frames_bias = cfg.MODEL.LSTR.LONG_MEMORY_UNMASKED_FRAMES_BIAS
        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            seed = np.random.randint(self.work_memory_length) if self.training else 0
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.work_memory_length),
                range(seed + self.work_memory_length, target.shape[0], self.work_memory_length)):
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end],
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target = self.inputs[index]

        visual_inputs = np.load( # load visual features with memory mapping to save memory
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load( # load motion features with memory mapping to save memory
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0) # get indices for work memory
        work_indices = work_indices[::self.work_memory_sample_rate] # subsample indices for work memory
        work_visual_inputs = visual_inputs[work_indices] # get visual features for work memory
        work_motion_inputs = motion_inputs[work_indices] # get motion features for work memory

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')

            # Mask a specific frame once it has arrived in the long memory window
            if self.long_memory_mask_frame >= 0 and work_start > self.long_memory_mask_frame:
                mask_pos = np.where(long_indices == self.long_memory_mask_frame)[0]
                if len(mask_pos) > 0:
                    memory_key_padding_mask[mask_pos] = float('-inf')

            # Randomly mask a fraction of valid (non-padded) long memory frames
            if self.long_memory_mask_ratio > 0.0:
                valid = np.where(memory_key_padding_mask == 0)[0]
                n = int(len(valid) * self.long_memory_mask_ratio)
                if n > 0:
                    masked = np.random.choice(valid, size=n, replace=False)
                    memory_key_padding_mask[masked] = self.long_memory_masked_frames_bias
                    unmasked = np.setdiff1d(valid, masked)
                    memory_key_padding_mask[unmasked] = self.long_memory_unmasked_frames_bias
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target
        else:
            return fusion_visual_inputs, fusion_motion_inputs, target

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRBatchInferenceTHUMOS')
@registry.register('LSTRBatchInferenceTVSeries')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.long_memory_mask_frame = cfg.MODEL.LSTR.LONG_MEMORY_MASK_FRAME
        self.long_memory_mask_ratio = cfg.MODEL.LSTR.LONG_MEMORY_MASK_RATIO
        self.long_memory_masked_frames_bias = cfg.MODEL.LSTR.LONG_MEMORY_MASKED_FRAMES_BIAS
        self.long_memory_unmasked_frames_bias = cfg.MODEL.LSTR.LONG_MEMORY_UNMASKED_FRAMES_BIAS

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            for work_start, work_end in zip(
                range(0, target.shape[0] + 1),
                range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end], target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0) # get indices for work memory
        work_indices = work_indices[::self.work_memory_sample_rate] # resample indices for work memory
        
        # NOTE: Getting work memory features (short-term memory) for the current segment. This is done for each segment in the batch inference setting, which allows us to leverage the short-term memory for better predictions.
        work_visual_inputs = visual_inputs[work_indices] # get visual features for work memory 
        work_motion_inputs = motion_inputs[work_indices] # get motion features for work memory

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')

            # Mask a specific frame once it has arrived in the long memory window
            if self.long_memory_mask_frame >= 0 and work_start > self.long_memory_mask_frame:
                mask_pos = np.where(long_indices == self.long_memory_mask_frame)[0]
                if len(mask_pos) > 0:
                    memory_key_padding_mask[mask_pos] = float('-inf')

            # Randomly mask a fraction of valid (non-padded) long memory frames
            if self.long_memory_mask_ratio > 0.0:
                valid = np.where(memory_key_padding_mask == 0)[0]
                n = int(len(valid) * self.long_memory_mask_ratio)
                if n > 0:
                    masked = np.random.choice(valid, size=n, replace=False)
                    memory_key_padding_mask[masked] = self.long_memory_masked_frames_bias
                    unmasked = np.setdiff1d(valid, masked)
                    memory_key_padding_mask[unmasked] = self.long_memory_unmasked_frames_bias
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)
