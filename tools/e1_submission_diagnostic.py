#!/usr/bin/env python3
"""Read-only E1 val diagnosis for the competition max-100 submission cap."""

from __future__ import print_function

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np

from e1_diagnostic_common import (REPO_ROOT, build_eval_trainer,
                                  decode_raw_query_boxes, load_annotation_maps,
                                  records_from_stage2, run_decoder_and_postprocess,
                                  runtime_info, stage2_rows_by_image, summary_stats,
                                  tensor_to_numpy)


DEFAULT_CONFIG = (
    'configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml')
CONFIDENCES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
NMS_VALUES = (None, 0.90, 0.80, 0.70, 0.60, 0.50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('output/diagnostics/e1_submission'))
    parser.add_argument('--device', default='gpu:0')
    return parser.parse_args()


def git_text(command):
    return subprocess.check_output(command, cwd=str(REPO_ROOT), text=True).strip()


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def xywh_to_xyxy(box):
    return np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                      dtype=np.float64)


def iou_xyxy(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_a + area_b - intersection, 1e-12)
    return intersection / union


def class_aware_nms(rows, threshold):
    if threshold is None:
        return sorted(rows, key=lambda item: (-item['score'], item['query_rank']))
    by_class = {}
    for row in rows:
        by_class.setdefault(row['class_id'], []).append(row)
    kept = []
    for class_rows in by_class.values():
        ordered = sorted(class_rows, key=lambda item: (-item['score'], item['query_rank']))
        kept_class = []
        for candidate in ordered:
            candidate_box = xywh_to_xyxy(candidate['bbox'])
            kept_boxes = [xywh_to_xyxy(item['bbox']) for item in kept_class]
            overlaps = iou_xyxy(candidate_box, kept_boxes)
            if not np.any(overlaps > threshold):
                kept_class.append(candidate)
        kept.extend(kept_class)
    return sorted(kept, key=lambda item: (-item['score'], item['query_rank']))


def sanitize_submission_rows(rows, image_info):
    width = float(image_info['width'])
    height = float(image_info['height'])
    sanitized = []
    for row in rows:
        x1, y1, width_box, height_box = row['bbox']
        x1 = min(max(float(x1), 0.0), width)
        y1 = min(max(float(y1), 0.0), height)
        x2 = min(max(float(x1 + width_box), 0.0), width)
        y2 = min(max(float(y1 + height_box), 0.0), height)
        if x2 <= x1 or y2 <= y1:
            continue
        copied = dict(row)
        copied['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        sanitized.append(copied)
    return sanitized


def submission_filter(rows, confidence, nms_threshold, image_info):
    sanitized = sanitize_submission_rows(rows, image_info)
    filtered = [item for item in sanitized if item['score'] >= confidence]
    return class_aware_nms(filtered, nms_threshold)[:100]


def coco_records(rows):
    return [{key: item[key] for key in ('image_id', 'category_id', 'bbox', 'score')}
            for item in rows]


def evaluate_coco(annotation_path, records, category_ids, class_names):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(coco_records(records))
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    with redirect_stdout(io.StringIO()):
        evaluator.summarize()
    stats = evaluator.stats
    precision = evaluator.eval['precision']
    recall = evaluator.eval['recall']
    per_class = []
    gt_by_cat = {cat_id: len(coco_gt.getAnnIds(catIds=[cat_id], iscrowd=False))
                 for cat_id in category_ids}
    for index, cat_id in enumerate(category_ids):
        category_precision = precision[:, :, index, 0, -1]
        valid = category_precision[category_precision > -1]
        ap = float(valid.mean()) if valid.size else float('nan')
        ap50_values = precision[0, :, index, 0, -1]
        ap50_values = ap50_values[ap50_values > -1]
        ap75_values = precision[5, :, index, 0, -1]
        ap75_values = ap75_values[ap75_values > -1]
        category_recall = recall[:, index, 0, -1]
        category_recall = category_recall[category_recall > -1]
        per_class.append({
            'class_id': index, 'category_id': cat_id,
            'class_name': class_names[index], 'gt_count': gt_by_cat[cat_id],
            'AP': ap,
            'AP50': float(ap50_values.mean()) if ap50_values.size else float('nan'),
            'AP75': float(ap75_values.mean()) if ap75_values.size else float('nan'),
            'Recall100': float(category_recall.mean()) if category_recall.size else float('nan'),
        })
    return ({'AP': float(stats[0]), 'AP50': float(stats[1]),
             'AP75': float(stats[2]), 'APS': float(stats[3]),
             'APM': float(stats[4]), 'APL': float(stats[5]),
             'AR100': float(stats[8])}, per_class)


def duplicate_counts(rows_by_image, thresholds, class_count):
    output = {threshold: {'pairs': 0, 'predictions': set(),
                          'per_class_pairs': [0] * class_count,
                          'per_class_predictions': [set() for _ in range(class_count)]}
              for threshold in thresholds}
    for rows in rows_by_image.values():
        for class_id in range(class_count):
            indices = [index for index, row in enumerate(rows)
                       if row['class_id'] == class_id]
            for left_pos, left_index in enumerate(indices):
                left_box = xywh_to_xyxy(rows[left_index]['bbox'])
                right_indices = indices[left_pos + 1:]
                right_boxes = [xywh_to_xyxy(rows[item]['bbox'])
                               for item in right_indices]
                overlaps = iou_xyxy(left_box, right_boxes)
                for right_index, overlap in zip(right_indices, overlaps):
                    for threshold in thresholds:
                        if overlap > threshold:
                            info = output[threshold]
                            info['pairs'] += 1
                            info['predictions'].update((id(rows[left_index]), id(rows[right_index])))
                            info['per_class_pairs'][class_id] += 1
                            info['per_class_predictions'][class_id].update(
                                (id(rows[left_index]), id(rows[right_index])))
    for threshold in thresholds:
        info = output[threshold]
        info['predictions'] = len(info['predictions'])
        info['per_class_predictions'] = [len(item) for item in info['per_class_predictions']]
    return output


def match_gt(rows, annotations, class_by_cat_id, threshold):
    unmatched = {index: True for index in range(len(annotations))}
    matched = set()
    for row in sorted(rows, key=lambda item: (-item['score'], item['query_rank'])):
        prediction_box = xywh_to_xyxy(row['bbox'])
        candidates = [index for index, annotation in enumerate(annotations)
                      if unmatched[index] and
                      class_by_cat_id[int(annotation['category_id'])] == row['class_id']]
        candidate_boxes = [xywh_to_xyxy(annotations[index]['bbox'])
                           for index in candidates]
        overlaps = iou_xyxy(prediction_box, candidate_boxes)
        if overlaps.size and overlaps.max() >= threshold:
            matched_index = candidates[int(overlaps.argmax())]
            unmatched[matched_index] = False
            matched.add(matched_index)
    return matched


def lost_top100(rows_by_image, final_by_image, annotations_by_image,
                class_by_cat_id, class_count, threshold):
    lost_per_class = [0] * class_count
    affected_images = set()
    for image_id, rows in rows_by_image.items():
        annotations = annotations_by_image.get(image_id, [])
        top100 = final_by_image[image_id]
        matched_top100 = match_gt(top100, annotations, class_by_cat_id, threshold)
        for gt_index, annotation in enumerate(annotations):
            if gt_index in matched_top100:
                continue
            class_id = class_by_cat_id[int(annotation['category_id'])]
            gt_box = xywh_to_xyxy(annotation['bbox'])
            candidates = [row for row in rows[100:] if row['class_id'] == class_id]
            candidate_boxes = [xywh_to_xyxy(row['bbox']) for row in candidates]
            overlaps = iou_xyxy(gt_box, candidate_boxes)
            if overlaps.size and overlaps.max() >= threshold:
                lost_per_class[class_id] += 1
                affected_images.add(image_id)
    return lost_per_class, affected_images


def per_class_occupancy(top100_by_image, class_count):
    values = [[] for _ in range(class_count)]
    for rows in top100_by_image.values():
        counts = [0] * class_count
        for row in rows:
            counts[row['class_id']] += 1
        for class_id in range(class_count):
            values[class_id].append(counts[class_id])
    return [summary_stats(item) for item in values]


def score_calibration(rows_by_image, annotations_by_image, class_by_cat_id,
                      class_count):
    all_scores = [[] for _ in range(class_count)]
    tp_scores = [[] for _ in range(class_count)]
    fp_scores = [[] for _ in range(class_count)]
    for image_id, rows in rows_by_image.items():
        annotations = annotations_by_image.get(image_id, [])
        matched = match_gt(rows, annotations, class_by_cat_id, 0.5)
        matched_rows = set()
        for class_id in range(class_count):
            class_rows = [row for row in rows if row['class_id'] == class_id]
            for row in class_rows:
                all_scores[class_id].append(row['score'])
                row_box = xywh_to_xyxy(row['bbox'])
                gt_indices = [index for index, annotation in enumerate(annotations)
                              if class_by_cat_id[int(annotation['category_id'])] == class_id]
                overlaps = iou_xyxy(row_box, [xywh_to_xyxy(annotations[index]['bbox'])
                                              for index in gt_indices])
                if overlaps.size and overlaps.max() >= 0.5:
                    tp_scores[class_id].append(row['score'])
                else:
                    fp_scores[class_id].append(row['score'])
    return ([summary_stats(item) for item in all_scores],
            [summary_stats(item) for item in tp_scores],
            [summary_stats(item) for item in fp_scores])


def write_report(path, args, git_before, baseline_metrics, per_class,
                 saturation, duplicates_raw, duplicates_top100, lost50, lost75,
                 occupancy, sweep_rows, equivalence_checks, environment):
    lines = ['# E1 Submission / Top-100 Diagnostic', '', '## A. Git / environment', '',
             '```text', 'branch: {}'.format(git_text(['git', 'branch', '--show-current'])),
             'HEAD: {}'.format(git_text(['git', 'log', '-1', '--oneline'])),
             'git status before: {}'.format(git_before or '(clean)'),
             'git status after: {}'.format(git_text(['git', 'status', '--short']) or '(unchanged)'),
             'config: {}'.format(Path(args.config).resolve()),
             'checkpoint: {}'.format(args.checkpoint.resolve()),
             'GPU: {}'.format(environment['gpu']),
             'Paddle: {}'.format(environment['paddle']),
             'CUDA: {}'.format(environment['cuda']),
             'device: {}'.format(args.device), '```', '', '## B. E1 baseline', '',
             '| AP | AP50 | AP75 | APS | APM | APL | AR100 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
             '| {AP:.6f} | {AP50:.6f} | {AP75:.6f} | {APS:.6f} | {APM:.6f} | {APL:.6f} | {AR100:.6f} |'.format(**baseline_metrics),
             '', 'Production equivalence: Stage-2 diagnostic predictions matched '
             'the current `model(batch)` output for {} images with coordinate '
             'tolerance `1e-5`; only then was the offline sweep run.'.format(
                 equivalence_checks), '',
             '| Class | GT | AP | AP50 | AP75 | Recall@100 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in per_class:
        lines.append('| {class_name} | {gt_count} | {AP:.6f} | {AP50:.6f} | '
                     '{AP75:.6f} | {Recall100:.6f} |'.format(**item))
    lines.extend([
             '', '## F. Top-100 saturation', '',
             '- Images final count ==100: {}/{} ({:.2%})'.format(
                 saturation['at100'], saturation['images'], saturation['at100'] / saturation['images']),
             '- Stage-2 raw >100: {}/{}; >150: {}/{}; >200: {}/{}'.format(
                 saturation['raw_gt100'], saturation['images'], saturation['raw_gt150'], saturation['images'],
                 saturation['raw_gt200'], saturation['images']), '',
             '## G. Duplicate stats', ''])
    for label, values in (('Raw300', duplicates_raw), ('Top100', duplicates_top100)):
        lines.append('### {}'.format(label))
        lines.append('')
        lines.append('| IoU | Duplicate pairs | Involved predictions |')
        lines.append('| ---: | ---: | ---: |')
        for threshold in (0.5, 0.7, 0.9):
            info = values[threshold]
            lines.append('| >{:.1f} | {} | {} |'.format(
                threshold, info['pairs'], info['predictions']))
        lines.append('')
    lines.append('Per-class duplicate details are in `occupancy_and_score_stats.csv`.')
    lines.extend(['## H. Rank >100 lost TP', '',
                  '| Class | Lost TP@50 | Lost TP@75 |',
                  '| --- | ---: | ---: |'])
    for item, value50, value75 in zip(per_class, lost50[0], lost75[0]):
        lines.append('| {} | {} | {} |'.format(item['class_name'], value50, value75))
    lines.append('Affected images: IoU@50={}, IoU@75={}.'.format(
        len(lost50[1]), len(lost75[1])))
    lines.extend(['', '## I. Top100 class occupancy', '',
                  '| Class | Mean/image | p50 | p90 | max |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for item, values in zip(per_class, occupancy):
        lines.append('| {} | {:.4f} | {:.1f} | {:.1f} | {:.0f} |'.format(
            item['class_name'], values['mean'], values['p50'], values['p90'], values['max']))
    lines.extend(['', '## J. Offline sweep', '',
                  'All 36 combinations are in `sweep_results.csv`. Baseline is `conf=0.05, NMS=None`.',
                  '', '## K. Evidence-only decision', ''])
    baseline_row = next(item for item in sweep_rows
                        if item['confidence'] == 0.05 and item['nms_iou'] == 'None')
    best_ap = max(sweep_rows, key=lambda item: item['AP'])
    best_aps = max(sweep_rows, key=lambda item: item['APS'])
    lines.append('- Q1/Q2: padding-aware mechanisms are not evaluated by this '
                 'submission-only script; use the companion padding report.')
    lines.append('- Q3: baseline AP={:.6f}; best offline AP={:.6f} at conf={}, NMS={}; '
                 'delta={:+.6f}. Best APS={:.6f} at conf={}, NMS={}. Any selected '
                 'setting has validation-overfit risk and is not applied automatically.'.format(
                     baseline_row['AP'], best_ap['AP'], best_ap['confidence'], best_ap['nms_iou'],
                     best_ap['AP'] - baseline_row['AP'], best_aps['APS'],
                     best_aps['confidence'], best_aps['nms_iou']))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    import paddle

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git_before = git_text(['git', 'status', '--short'])
    cfg, trainer = build_eval_trainer(args.config, args.checkpoint, args.device)
    environment = runtime_info(args.device)
    annotation_path = REPO_ROOT / cfg.EvalDataset['anno_path']
    (_, image_by_id, annotations_by_image, cat_id_by_class, class_names,
     class_by_cat_id) = load_annotation_maps(annotation_path)
    all_stage2 = []
    raw_query_ids = []
    raw_query_labels = []
    raw_query_scores = []
    raw_query_boxes = []
    equivalence_checks = 0
    with paddle.no_grad():
        for step_id, batch in enumerate(trainer.loader):
            with paddle.amp.auto_cast(enable=True):
                raw_boxes, raw_logits, stage2_boxes, stage2_num = run_decoder_and_postprocess(
                    trainer.model, batch)
                raw_boxes_original = decode_raw_query_boxes(
                    raw_boxes, batch['im_shape'], batch['scale_factor'],
                    paddle.shape(batch['vis_image'])[2:])
            if step_id < 10:
                with paddle.amp.auto_cast(enable=True):
                    production = trainer.model(batch)
                if not np.allclose(tensor_to_numpy(production['bbox']),
                                   tensor_to_numpy(stage2_boxes), rtol=1e-5,
                                   atol=1e-5):
                    raise RuntimeError('Stage-2 dump mismatch with production output at step {}'.format(step_id))
                if not np.array_equal(tensor_to_numpy(production['bbox_num']),
                                      tensor_to_numpy(stage2_num)):
                    raise RuntimeError('Stage-2 count mismatch with production output')
                equivalence_checks += 1
            image_ids = tensor_to_numpy(batch['im_id']).astype(np.int64).reshape(-1)
            raw_logits_np = tensor_to_numpy(raw_logits)
            raw_boxes_np = tensor_to_numpy(raw_boxes_original)
            raw_labels_np = raw_logits_np.argmax(axis=-1).astype(np.int64)
            raw_scores_np = 1.0 / (1.0 + np.exp(-raw_logits_np.max(axis=-1)))
            raw_query_ids.extend(image_ids.tolist())
            raw_query_labels.extend(raw_labels_np.tolist())
            raw_query_scores.extend(raw_scores_np.tolist())
            raw_query_boxes.extend(raw_boxes_np.tolist())
            all_stage2.extend(records_from_stage2(
                stage2_boxes, stage2_num, batch['im_id'], cat_id_by_class))
            if (step_id + 1) % 25 == 0 or step_id + 1 == len(trainer.loader):
                print('Submission diagnostic: {}/{} images'.format(
                    step_id + 1, len(trainer.loader)), flush=True)
    if len(set(record['image_id'] for record in all_stage2)) != 400:
        raise RuntimeError('Stage-2 dump does not contain all 400 validation images')
    if equivalence_checks != 10:
        raise RuntimeError('Expected 10 production equivalence checks, got {}'.format(
            equivalence_checks))
    (args.output_dir / 'raw_predictions.json').write_text(
        json.dumps(all_stage2, ensure_ascii=False), encoding='utf-8')
    (args.output_dir / 'final_predictions_baseline.json').write_text(
        json.dumps(coco_records(all_stage2), ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.output_dir / 'raw_decoder_queries.npz',
                        image_id=np.asarray(raw_query_ids, dtype=np.int64),
                        class_id=np.asarray(raw_query_labels, dtype=np.int64),
                        score=np.asarray(raw_query_scores, dtype=np.float32),
                        bbox_xyxy=np.asarray(raw_query_boxes, dtype=np.float32))
    metrics, per_class = evaluate_coco(
        annotation_path, all_stage2, list(cat_id_by_class.values()), class_names)
    write_csv(args.output_dir / 'per_class_stats.csv', per_class,
              ['class_id', 'category_id', 'class_name', 'gt_count',
               'AP', 'AP50', 'AP75', 'Recall100'])
    stage2_by_image = stage2_rows_by_image(all_stage2)
    baseline_top100_by_image = {
        image_id: submission_filter(rows, 0.05, None, image_by_id[image_id])
        for image_id, rows in stage2_by_image.items()}
    baseline_top100 = [row for rows in baseline_top100_by_image.values()
                       for row in rows]
    (args.output_dir / 'submission_baseline_top100.json').write_text(
        json.dumps(coco_records(baseline_top100), ensure_ascii=False), encoding='utf-8')
    per_image_rows = []
    for image_id, rows in stage2_by_image.items():
        final_rows = baseline_top100_by_image[image_id]
        per_image_rows.append({
            'image_id': image_id, 'file_name': image_by_id[image_id]['file_name'],
            'gt_count': len(annotations_by_image.get(image_id, [])),
            'raw_prediction_count': len(rows), 'final_prediction_count': len(final_rows),
            'final_count_eq100': int(len(final_rows) == 100)})
    write_csv(args.output_dir / 'per_image_stats.csv', per_image_rows,
              ['image_id', 'file_name', 'gt_count', 'raw_prediction_count',
               'final_prediction_count', 'final_count_eq100'])
    saturation = {
        'images': len(per_image_rows),
        'at100': sum(row['final_count_eq100'] for row in per_image_rows),
        'raw_gt100': sum(row['raw_prediction_count'] > 100 for row in per_image_rows),
        'raw_gt150': sum(row['raw_prediction_count'] > 150 for row in per_image_rows),
        'raw_gt200': sum(row['raw_prediction_count'] > 200 for row in per_image_rows),
    }
    class_count = len(cat_id_by_class)
    duplicates_raw = duplicate_counts(stage2_by_image, (0.5, 0.7, 0.9), class_count)
    duplicates_top100 = duplicate_counts(
        baseline_top100_by_image, (0.5, 0.7, 0.9), class_count)
    lost50 = lost_top100(stage2_by_image, baseline_top100_by_image,
                         annotations_by_image, class_by_cat_id, class_count, 0.5)
    lost75 = lost_top100(stage2_by_image, baseline_top100_by_image,
                         annotations_by_image, class_by_cat_id, class_count, 0.75)
    occupancy = per_class_occupancy(baseline_top100_by_image, class_count)
    all_score, tp_score, fp_score = score_calibration(
        stage2_by_image, annotations_by_image, class_by_cat_id, class_count)
    occupancy_rows = []
    for class_id, values in enumerate(occupancy):
        occupancy_rows.append({
            'class_id': class_id, 'class_name': class_names[class_id], **values,
            'score_all_mean': all_score[class_id]['mean'],
            'score_tp50_mean': tp_score[class_id]['mean'],
            'score_fp50_mean': fp_score[class_id]['mean'],
            'raw_duplicate_pairs_iou50': duplicates_raw[0.5]['per_class_pairs'][class_id],
            'top100_duplicate_pairs_iou50': duplicates_top100[0.5]['per_class_pairs'][class_id],
            'lost_tp50': lost50[0][class_id], 'lost_tp75': lost75[0][class_id]})
    write_csv(args.output_dir / 'occupancy_and_score_stats.csv', occupancy_rows,
              list(occupancy_rows[0].keys()))
    sweep_rows = []
    for confidence in CONFIDENCES:
        for nms_threshold in NMS_VALUES:
            selected = [item for rows in stage2_by_image.values()
                        for item in submission_filter(
                            rows, confidence, nms_threshold, image_by_id[image_id])]
            sweep_metrics, _ = evaluate_coco(
                annotation_path, selected, list(cat_id_by_class.values()), class_names)
            sweep_rows.append({'confidence': confidence,
                               'nms_iou': 'None' if nms_threshold is None else nms_threshold,
                               **sweep_metrics,
                               'prediction_count': len(selected)})
            print('Sweep conf={} nms={} AP={:.6f}'.format(
                confidence, nms_threshold, sweep_metrics['AP']), flush=True)
    write_csv(args.output_dir / 'sweep_results.csv', sweep_rows,
              ['confidence', 'nms_iou', 'AP', 'AP50', 'AP75', 'APS', 'APM',
               'APL', 'AR100', 'prediction_count'])
    diagnostic_summary = {
        'production_equivalence_checks': equivalence_checks,
        'baseline_evaluator': metrics, 'saturation': saturation,
        'lost_tp50': lost50[0], 'lost_tp75': lost75[0],
        'affected_images_tp50': len(lost50[1]),
        'affected_images_tp75': len(lost75[1]),
        'duplicate_raw': duplicates_raw, 'duplicate_top100': duplicates_top100,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(diagnostic_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(args.output_dir / 'report.md', args, git_before, metrics,
                 per_class, saturation, duplicates_raw, duplicates_top100,
                 lost50, lost75, occupancy, sweep_rows, equivalence_checks,
                 environment)
    print('Submission diagnostic complete: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
