#!/usr/bin/env python3
"""Run tri-modal DAMSDet inference and build the exact AIC2026 TXT ZIP."""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-w', '--weights', required=True)
    parser.add_argument('--vis-dir', required=True, type=Path)
    parser.add_argument('--ir-dir', required=True, type=Path)
    parser.add_argument('--depth-dir', required=True, type=Path)
    parser.add_argument('--output-dir',
                        type=Path,
                        default=Path('submission_aic2026'))
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument(
        '--class-thresholds',
        type=Path,
        help='Optional JSON object/list of validation-calibrated thresholds.')
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.0,
        help='Optional class-aware NMS IoU; 0 disables NMS for DETR output.')
    parser.add_argument('--max-detections', type=int, default=100)
    parser.add_argument('--device', default='gpu:0')
    return parser.parse_args()


def aligned_images(vis_dir, ir_dir, depth_dir):
    vis = sorted(p for p in vis_dir.iterdir()
                 if p.suffix.lower() in IMAGE_SUFFIXES)
    ir_by_name = {
        p.name: p
        for p in ir_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    depth_by_name = {
        p.name: p
        for p in depth_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = [
        p.name for p in vis
        if p.name not in ir_by_name or p.name not in depth_by_name
    ]
    if missing:
        raise RuntimeError('Unaligned modalities, first missing: {}'.format(
            missing[:5]))
    return (vis, [ir_by_name[p.name]
                  for p in vis], [depth_by_name[p.name] for p in vis])


def write_prediction(path, rows):
    lines = [
        '{} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}'.format(*row) for row in rows
    ]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''),
                    encoding='utf-8')


def load_class_thresholds(path):
    if path is None:
        return None
    values = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(values, list):
        if len(values) != 12:
            raise ValueError('Class-threshold list must contain 12 values.')
        thresholds = {index: float(value)
                      for index, value in enumerate(values)}
    elif isinstance(values, dict):
        thresholds = {int(key): float(value) for key, value in values.items()}
    else:
        raise ValueError('Class thresholds must be a JSON object or list.')
    if any(class_id < 0 or class_id >= 12
           for class_id in thresholds):
        raise ValueError('Class-threshold keys must be in [0, 11].')
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0
           for value in thresholds.values()):
        raise ValueError('Class thresholds must be in [0, 1].')
    return thresholds


def class_aware_nms(boxes, iou_threshold):
    """Return class-aware NMS output in descending confidence order."""
    if len(boxes) == 0 or iou_threshold <= 0.0:
        return boxes[np.argsort(-boxes[:, 1])] if len(boxes) else boxes
    kept = []
    for class_id in np.unique(boxes[:, 0].astype(np.int64)):
        class_indices = np.flatnonzero(boxes[:, 0].astype(np.int64) == class_id)
        order = class_indices[np.argsort(-boxes[class_indices, 1])]
        while len(order):
            current = order[0]
            kept.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[current, 2], boxes[rest, 2])
            y1 = np.maximum(boxes[current, 3], boxes[rest, 3])
            x2 = np.minimum(boxes[current, 4], boxes[rest, 4])
            y2 = np.minimum(boxes[current, 5], boxes[rest, 5])
            intersection = np.maximum(0.0, x2 - x1) * np.maximum(
                0.0, y2 - y1)
            current_area = ((boxes[current, 4] - boxes[current, 2]) *
                            (boxes[current, 5] - boxes[current, 3]))
            rest_area = ((boxes[rest, 4] - boxes[rest, 2]) *
                         (boxes[rest, 5] - boxes[rest, 3]))
            union = np.maximum(current_area + rest_area - intersection, 1e-9)
            order = rest[intersection / union <= iou_threshold]
    kept = np.asarray(kept, dtype=np.int64)
    return boxes[kept[np.argsort(-boxes[kept, 1])]]


def build_txt_files(results,
                    vis_images,
                    txt_dir,
                    threshold,
                    max_detections,
                    class_thresholds=None,
                    nms_iou=0.0):
    cursor = 0
    for outs in results:
        boxes = np.asarray(outs['bbox']).reshape(-1, 6)
        counts = np.asarray(outs['bbox_num']).reshape(-1)
        image_ids = np.asarray(outs['im_id']).reshape(-1)
        start = 0
        for count, image_id in zip(counts, image_ids):
            image_path = vis_images[int(image_id)]
            with Image.open(image_path) as image:
                width, height = image.size
            selected = boxes[start:start + int(count)]
            start += int(count)
            if len(selected):
                selected = selected.copy()
                selected[:, [2, 4]] = np.clip(selected[:, [2, 4]], 0.0,
                                              float(width))
                selected[:, [3, 5]] = np.clip(selected[:, [3, 5]], 0.0,
                                              float(height))
                valid_boxes = np.logical_and(selected[:, 4] > selected[:, 2],
                                             selected[:, 5] > selected[:, 3])
                selected = selected[valid_boxes]
            if class_thresholds:
                thresholds = np.asarray([
                    class_thresholds.get(int(class_id), threshold)
                    for class_id in selected[:, 0]
                ])
                selected = selected[selected[:, 1] >= thresholds]
            else:
                selected = selected[selected[:, 1] >= threshold]
            selected = class_aware_nms(selected, nms_iou)
            selected = selected[np.argsort(-selected[:, 1])[:max_detections]]
            rows = []
            for class_id, score, x1, y1, x2, y2 in selected:
                cx = ((x1 + x2) * 0.5) / width
                cy = ((y1 + y2) * 0.5) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                rows.append((int(class_id), cx, cy, bw, bh, float(score)))
            write_prediction(txt_dir / (image_path.stem + '.txt'), rows)
            cursor += 1
    if cursor != len(vis_images):
        raise RuntimeError('Inference returned {} of {} images.'.format(
            cursor, len(vis_images)))


def validate(txt_dir, expected_stems, max_detections):
    files = sorted(txt_dir.glob('*.txt'))
    if {p.stem for p in files} != set(expected_stems):
        raise RuntimeError('Submission filenames do not match the test set.')
    for path in files:
        lines = [
            line for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        if len(lines) > max_detections:
            raise RuntimeError('{} has more than {} boxes.'.format(
                path.name, max_detections))
        for line_no, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise RuntimeError('{}:{} must have six fields.'.format(
                    path.name, line_no))
            class_id = int(fields[0])
            values = list(map(float, fields[1:]))
            if not 0 <= class_id < 12:
                raise RuntimeError('{}:{} invalid class.'.format(
                    path.name, line_no))
            if not all(0.0 <= value <= 1.0 for value in values):
                raise RuntimeError('{}:{} value outside [0,1].'.format(
                    path.name, line_no))


def main():
    args = parse_args()
    if args.score_threshold < 0.0 or args.score_threshold > 1.0:
        raise ValueError('--score-threshold must be in [0, 1].')
    if args.max_detections < 1 or args.max_detections > 100:
        raise ValueError('--max-detections must be in [1, 100].')
    if args.nms_iou < 0.0 or args.nms_iou > 1.0:
        raise ValueError('--nms-iou must be in [0, 1].')
    class_thresholds = load_class_thresholds(args.class_thresholds)
    vis, ir, depth = aligned_images(args.vis_dir.resolve(),
                                    args.ir_dir.resolve(),
                                    args.depth_dir.resolve())
    if len(vis) != 1000:
        raise RuntimeError('Expected 1000 test images, found {}.'.format(
            len(vis)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = args.output_dir / 'predictions'
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir()

    import paddle
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer

    paddle.set_device(args.device)
    cfg = load_config(args.config)
    cfg.weights = args.weights
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(args.weights)
    results = trainer.multi_predict([str(p) for p in vis],
                                    [str(p) for p in ir],
                                    [str(p) for p in depth],
                                    save_results=False,
                                    visualize=False)
    build_txt_files(results, vis, txt_dir, args.score_threshold,
                    args.max_detections, class_thresholds, args.nms_iou)
    validate(txt_dir, [p.stem for p in vis], args.max_detections)

    zip_path = args.output_dir / 'predictions.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(txt_dir.glob('*.txt')):
            archive.write(path, path.name)
    print('Submission ready: {}'.format(zip_path.resolve()))


if __name__ == '__main__':
    main()
