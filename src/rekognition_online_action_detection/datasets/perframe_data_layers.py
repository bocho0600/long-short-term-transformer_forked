# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

import torch
import torch.utils.data as data
import numpy as np

from rekognition_online_action_detection.utils.lstr_memory import (
    memory_key_padding_mask_from_indices,
    resolve_long_memory_window,
    segment_sampler,
    uniform_sampler,
)

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
        self.long_memory_window_length = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_LENGTH
        self.long_memory_window_stride = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_STRIDE
        self.long_memory_window_index = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_INDEX
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
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

    def __getitem__(self, index):
        session, work_start, work_end, target = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = resolve_long_memory_window(
                work_start,
                self.long_memory_length,
                self.long_memory_window_length,
                self.long_memory_window_stride,
                self.long_memory_window_index)
            if self.training:
                long_indices = segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = memory_key_padding_mask_from_indices(long_indices)
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
        self.long_memory_window_length = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_LENGTH
        self.long_memory_window_stride = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_STRIDE
        self.long_memory_window_index = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_INDEX
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

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

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = resolve_long_memory_window(
                work_start,
                self.long_memory_length,
                self.long_memory_window_length,
                self.long_memory_window_stride,
                self.long_memory_window_index)
            long_indices = uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = memory_key_padding_mask_from_indices(long_indices)
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
