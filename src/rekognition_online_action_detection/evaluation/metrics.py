# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from multiprocessing import Pool
from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def perframe_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result['per_class_AP'][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result


def get_labels_start_end_time(frame_wise_labels, bg_class):
    """Collapse a per-frame label sequence into action segments, dropping
    background/ignore frames. Returns the segment labels and their
    (start, end) frame boundaries. Adapted from the MS-TCN evaluation code
    (Farha & Gall, 2019).
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels))
    return labels, starts, ends


def levenstein(p, y, norm=False):
    """Levenshtein (edit) distance between two segment-label sequences."""
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    D[:, 0] = np.arange(m_row + 1)
    D[0, :] = np.arange(n_col + 1)
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100 if max(m_row, n_col) > 0 else 100.0
    else:
        score = D[-1, -1]
    return score


def segmental_edit_score(recognized, ground_truth, bg_class):
    """Normalized (0-100) segmental edit score for a single sequence."""
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm=True)


def f_score_counts(recognized, ground_truth, overlap, bg_class):
    """True/false positive and false negative segment counts at a temporal
    IoU threshold `overlap`, for a single sequence.
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        if len(y_label) == 0:
            fp += 1
            continue
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        same_class = np.array([p_label[j] == y_label[x] for x in range(len(y_label))])
        IoU = (1.0 * intersection / union) * same_class
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - np.sum(hits)
    return float(tp), float(fp), float(fn)


def perframe_segment_scores(ground_truth,
                            prediction,
                            ignore_index,
                            postprocessing,
                            overlaps=(0.1, 0.25, 0.5),
                            background_index=0):
    """Compute segmentation-based metrics (segmental edit score and F1@k) from
    per-session per-frame class scores.

    Unlike the frame-level mAP, these metrics operate on the sequence of action
    *segments* within each video, so they reward temporal coherence and
    penalize over-/under-segmentation and mislocalized action<->background
    transitions.

    Args:
        ground_truth, prediction: dicts mapping session -> array of shape
            (num_frames, num_classes). One-hot (or score) per frame; the
            per-frame label is taken as the argmax over classes.
        ignore_index: class treated as background for segment extraction
            (e.g. THUMOS 'Ambiguous'), so it never forms an action segment.
        postprocessing: optional (ground_truth, prediction) -> (gt, pred)
            hook applied per session before taking argmax.
        overlaps: temporal-IoU thresholds for F1@k.
        background_index: class index treated as background (default 0).

    Returns:
        OrderedDict with 'edit' and 'F1@{k}' keys, plus per-session 'edit'
        scores under 'per_session_edit'.
    """
    bg_class = set([background_index, ignore_index])

    result = OrderedDict()
    result['per_session_edit'] = OrderedDict()

    edit_scores = []
    tp = {ov: 0.0 for ov in overlaps}
    fp = {ov: 0.0 for ov in overlaps}
    fn = {ov: 0.0 for ov in overlaps}

    for session in ground_truth:
        gt = np.array(ground_truth[session])
        pred = np.array(prediction[session])
        if postprocessing is not None:
            gt, pred = postprocessing(gt, pred)
        if gt.shape[0] == 0:
            continue

        gt_labels = np.argmax(gt, axis=1).tolist()
        pred_labels = np.argmax(pred, axis=1).tolist()

        edit = segmental_edit_score(pred_labels, gt_labels, bg_class)
        result['per_session_edit'][session] = edit
        edit_scores.append(edit)

        for ov in overlaps:
            tp_i, fp_i, fn_i = f_score_counts(pred_labels, gt_labels, ov, bg_class)
            tp[ov] += tp_i
            fp[ov] += fp_i
            fn[ov] += fn_i

    result['edit'] = float(np.mean(edit_scores)) if edit_scores else 0.0
    for ov in overlaps:
        precision = tp[ov] / (tp[ov] + fp[ov]) if (tp[ov] + fp[ov]) > 0 else 0.0
        recall = tp[ov] / (tp[ov] + fn[ov]) if (tp[ov] + fn[ov]) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result['F1@{:d}'.format(int(ov * 100))] = f1 * 100
    return result


def get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e):
    starts = []
    ends = []
    stage_gt_targets = []
    stage_pred_scores = []
    for i in range(len(gt_targets)):
        if gt_targets[i] == 0:
            stage_gt_targets.append(gt_targets[i])
            stage_pred_scores.append(pred_scores[i])
        else:
            if i == 0 or gt_targets[i - 1] == 0:
                starts.append(i)
            if i == len(gt_targets) - 1 or gt_targets[i + 1] == 0:
                ends.append(i)
    if len(starts) != len(ends):
        raise ValueError('starts and ends cannot pair!')

    action_lens = [ends[i] - starts[i] for i in range(len(starts))]
    stage_starts = [starts[i] + int(action_lens[i] * perc_s) for i in range(len(starts))]
    stage_ends = [max(stage_starts[i] + 1, starts[i] + int(action_lens[i] * perc_e)) for i in range(len(starts))]
    for i in range(len(starts)):
        stage_gt_targets.extend(gt_targets[stage_starts[i]: stage_ends[i]])
        stage_pred_scores.extend(pred_scores[stage_starts[i]: stage_ends[i]])
    return np.array(stage_gt_targets), np.array(stage_pred_scores)


def perstage_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    for perc_s in range(10):
        perc_e = perc_s + 1
        stage_name = '{:2}%_{:3}%'.format(perc_s * 10, perc_e * 10)
        result[stage_name] = OrderedDict({'per_class_AP': OrderedDict()})
        for idx, class_name in enumerate(class_names):
            if idx not in ignore_index:
                stage_gt_targets, stage_pred_scores = get_stage_pred_scores(
                    (ground_truth[:, idx] == 1).astype(int),
                    prediction[:, idx],
                    perc_s / 10,
                    perc_e / 10,
                )
                result[stage_name]['per_class_AP'][class_name] = \
                    compute_score(stage_gt_targets, stage_pred_scores)
        result[stage_name]['mean_AP'] = \
            np.mean(list(result[stage_name]['per_class_AP'].values()))

    return result
