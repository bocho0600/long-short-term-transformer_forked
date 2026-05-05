# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from tqdm import tqdm

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def _pred_scores_file(cfg):
    pred_scores_file = osp.splitext(cfg.MODEL.CHECKPOINT)[0]
    if cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_SECONDS > 0:
        stride = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_STRIDE_SECONDS
        if stride == 0:
            stride = cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_SECONDS
        pred_scores_file += '_longwin{}s_stride{}s_idx{}'.format(
            cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_SECONDS,
            stride,
            cfg.MODEL.LSTR.LONG_MEMORY_WINDOW_INDEX)
    return pred_scores_file + '.pkl'


def do_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE * 16,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    gt_targets = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]

            score = model(*[x.to(device) for x in data[:-4]])
            score = score.softmax(dim=-1).cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] == 0:
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

    # Save scores and targets
    pred_scores_file = _pred_scores_file(cfg)
    pkl.dump({
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
    }, open(pred_scores_file, 'wb'))
    logger.info('Saved prediction scores to {}'.format(pred_scores_file))

    # Compute results
    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))
