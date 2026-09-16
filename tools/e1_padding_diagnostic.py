#!/usr/bin/env python3
"""Read-only full-validation diagnosis of E1 padding in encoder Top-300."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from e1_diagnostic_common import (REPO_ROOT, build_eval_trainer,
                                  load_annotation_maps, make_feature_pad_masks,
                                  run_encoder_topk, runtime_info, summary_stats,
                                  tensor_to_numpy)


DEFAULT_CONFIG = (
    'configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml')
LEVEL_NAMES = ('Fused P3', 'Fused P4', 'Fused P5',
               'Fused P6', 'Fused P7', 'Fused P8')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('output/diagnostics/e1_padding'))
    parser.add_argument('--device', default='gpu:0')
    return parser.parse_args()


def git_text(command):
    return subprocess.check_output(command, cwd=str(REPO_ROOT), text=True).strip()


def level_name(index, shape, neck_shapes):
    spatial = (int(shape[0]), int(shape[1]))
    if index < len(neck_shapes) and spatial == neck_shapes[index]:
        return 'Fused P{}'.format(index + 3)
    return 'Fused extra L{} ({}x{})'.format(index, spatial[0], spatial[1])


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_stats(values):
    stats = summary_stats(values)
    return ('mean={mean:.6f}, std={std:.6f}, min={min:.6f}, p10={p10:.6f}, '
            'p50={p50:.6f}, p90={p90:.6f}, p95={p95:.6f}, max={max:.6f}').format(
                **stats)


def write_report(path, args, git_before, rows, level_totals, runtime_levels,
                 score_valid, score_padding, environment):
    input_padding = [row['input_padding_ratio'] for row in rows]
    topk_padding = [row['topk_padding'] for row in rows]
    topk_ratio = [row['topk_padding_ratio'] for row in rows]
    image_count = len(rows)
    nonzero = sum(value > 0 for value in topk_padding)
    over5 = sum(value > 0.05 for value in topk_ratio)
    over10 = sum(value > 0.10 for value in topk_ratio)
    over20 = sum(value > 0.20 for value in topk_ratio)
    valid_score = np.asarray(score_valid, dtype=np.float64)
    padding_score = np.asarray(score_padding, dtype=np.float64)
    lines = []
    lines.append('# E1 Padding → Top-300 Query Diagnostic')
    lines.append('')
    lines.append('## A. Git / environment')
    lines.append('')
    lines.append('```text')
    lines.append('branch: {}'.format(git_text(['git', 'branch', '--show-current'])))
    lines.append('HEAD: {}'.format(git_text(['git', 'log', '-1', '--oneline'])))
    lines.append('git status before: {}'.format(git_before or '(clean)'))
    lines.append('git status after: {}'.format(git_text(['git', 'status', '--short']) or '(unchanged)'))
    lines.append('config: {}'.format(Path(args.config).resolve()))
    lines.append('checkpoint: {}'.format(args.checkpoint.resolve()))
    lines.append('GPU: {}'.format(environment['gpu']))
    lines.append('Paddle: {}'.format(environment['paddle']))
    lines.append('CUDA: {}'.format(environment['cuda']))
    lines.append('device: {}'.format(args.device))
    lines.append('```')
    lines.append('')
    lines.append('## C. Padding geometry')
    lines.append('')
    lines.append('- Input padding ratio: {}'.format(markdown_stats(input_padding)))
    lines.append('- Runtime `pad_mask` semantics: min/max/mean are recorded per '
                 'image in the CSV. The diagnostic requires binary `1=real`, '
                 '`0=padding` before nearest feature downsampling.')
    lines.append('- Runtime memory ordering was verified from tensors, not assumed:')
    lines.append('')
    lines.append('| Index | Runtime source | H×W | Padding cells | Padding ratio |')
    lines.append('| ---: | --- | ---: | ---: | ---: |')
    for index, item in enumerate(runtime_levels):
        total = level_totals[index]['total_cells']
        padded = level_totals[index]['padding_cells']
        lines.append('| {} | {} | {}×{} | {} | {:.6%} |'.format(
            index, item['name'], item['height'], item['width'], padded,
            padded / max(total, 1)))
    lines.append('')
    lines.append('The runtime path is `visir_feats = vis_feats + ir_feats`; therefore '
                 'these are fused P3/P4/P5 plus transformer-generated extra '
                 'levels, not separate RGB then IR memory blocks.')
    lines.append('')
    lines.append('## D. Top-300 padding pollution')
    lines.append('')
    lines.append('- Padded queries/image: {}'.format(markdown_stats(topk_padding)))
    lines.append('- Padded-query ratio: {}'.format(markdown_stats(topk_ratio)))
    lines.append('- Images with >0 padding queries: {}/{} ({:.2%})'.format(
        nonzero, image_count, nonzero / max(image_count, 1)))
    lines.append('- Images with padding TopK >5%: {}/{} ({:.2%})'.format(
        over5, image_count, over5 / max(image_count, 1)))
    lines.append('- Images with padding TopK >10%: {}/{} ({:.2%})'.format(
        over10, image_count, over10 / max(image_count, 1)))
    lines.append('- Images with padding TopK >20%: {}/{} ({:.2%})'.format(
        over20, image_count, over20 / max(image_count, 1)))
    lines.append('')
    lines.append('| Level | TopK selections | Valid | Padding | Padding % |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for index, item in enumerate(runtime_levels):
        values = level_totals[index]
        lines.append('| {} | {} | {} | {} | {:.6%} |'.format(
            item['name'], values['topk'], values['topk_valid'],
            values['topk_padding'], values['topk_padding'] /
            max(values['topk'], 1)))
    lines.append('')
    lines.append('## E. Padding vs valid encoder TopK score')
    lines.append('')
    lines.append('- Valid selected logits: {}'.format(markdown_stats(valid_score)))
    lines.append('- Padding selected logits: {}'.format(markdown_stats(padding_score)))
    lines.append('- Valid selected sigmoid scores: {}'.format(
        markdown_stats(1.0 / (1.0 + np.exp(-valid_score)))))
    lines.append('- Padding selected sigmoid scores: {}'.format(
        markdown_stats(1.0 / (1.0 + np.exp(-padding_score)))))
    lines.append('')
    lines.append('## K. Evidence-only decision')
    lines.append('')
    mean_ratio = float(np.mean(topk_ratio)) if topk_ratio else 0.0
    if mean_ratio >= 0.05 or over10 > 0:
        verdict = ('Padding-aware TopK has a concrete ablation rationale: '
                   'the measured selected-padding rate is non-trivial.')
    else:
        verdict = ('Padding-aware TopK currently lacks a strong rationale: '
                   'selected padding is rare under the measured checkpoint.')
    lines.append('- Q1/Q2/Q3: Top-300 padding selection is quantified above; '
                 'mean ratio is {:.6%}.'.format(mean_ratio))
    lines.append('- Q4/Q5: source-level attribution is the runtime fused-level '
                 'table above; RGB/IR are fused before memory construction.')
    lines.append('- Q6: score distributions above distinguish low-score padding '
                 'from competitive padding candidates.')
    lines.append('- Q7: {}'.format(verdict))
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
    (_, image_by_id, annotations_by_image, _, _, _) = load_annotation_maps(
        annotation_path)
    rows = []
    level_totals = []
    runtime_levels = []
    score_valid = []
    score_padding = []
    observed_mapping = None
    production_topk_checks = 0
    with paddle.no_grad():
        for step_id, batch in enumerate(trainer.loader):
            with paddle.amp.auto_cast(enable=True):
                probe = run_encoder_topk(trainer.model, batch)
                level_masks, memory_valid = make_feature_pad_masks(
                    batch['pad_mask'], probe['spatial_shapes'])
                if step_id < 10:
                    production_out = trainer.model.transformer(
                        None, probe['vis_feats'], probe['ir_feats'],
                        batch.get('pad_mask'), batch)
                    production_enc_logits = production_out[3]
                    if not bool(paddle.allclose(
                            production_enc_logits,
                            probe['selected_logits'], rtol=1e-5,
                            atol=1e-6).numpy().item()):
                        raise RuntimeError(
                            'Reconstructed TopK does not match production '
                            'encoder selection for image {}'.format(step_id))
                    production_topk_checks += 1
            if memory_valid.shape[1] != probe['memory'].shape[1]:
                raise RuntimeError('memory_valid_mask length {} != memory length {}'.format(
                    memory_valid.shape[1], probe['memory'].shape[1]))
            shapes = [(int(item[0]), int(item[1]))
                      for item in probe['spatial_shapes']]
            neck_shapes = [(int(item.shape[-2]), int(item.shape[-1]))
                           for item in probe['vis_feats']]
            names = [level_name(index, shape, neck_shapes)
                     for index, shape in enumerate(shapes)]
            if observed_mapping is None:
                observed_mapping = (shapes, names)
                runtime_levels = [
                    {'name': name, 'height': shape[0], 'width': shape[1]}
                    for name, shape in zip(names, shapes)]
                level_totals = [
                    {'total_cells': 0, 'padding_cells': 0, 'topk': 0,
                     'topk_valid': 0, 'topk_padding': 0}
                    for _ in shapes]
            elif observed_mapping != (shapes, names):
                raise RuntimeError('Feature-level mapping changed inside eval loader')

            pad_mask = tensor_to_numpy(batch['pad_mask'])
            if pad_mask.min() < -1e-6 or pad_mask.max() > 1.0 + 1e-6:
                raise RuntimeError('pad_mask is outside the expected [0, 1] range')
            image_id = int(tensor_to_numpy(batch['im_id']).reshape(-1)[0])
            topk_indices = tensor_to_numpy(probe['topk_indices']).reshape(-1)
            topk_logits = tensor_to_numpy(probe['topk_logits']).reshape(-1)
            feature_masks = [tensor_to_numpy(item).reshape(-1).astype(bool)
                             for item in level_masks]
            level_offsets = np.cumsum([0] + [mask.size for mask in feature_masks])
            selected_padding = 0
            selected_by_level = []
            selected_padding_by_level = []
            for index, feature_mask in enumerate(feature_masks):
                level_totals[index]['total_cells'] += int(feature_mask.size)
                level_totals[index]['padding_cells'] += int((~feature_mask).sum())
                selected = ((topk_indices >= level_offsets[index]) &
                            (topk_indices < level_offsets[index + 1]))
                local_indices = topk_indices[selected] - level_offsets[index]
                local_valid = feature_mask[local_indices]
                topk_count = int(selected.sum())
                topk_valid = int(local_valid.sum())
                topk_padding = topk_count - topk_valid
                selected_by_level.append(topk_count)
                selected_padding_by_level.append(topk_padding)
                selected_padding += topk_padding
                level_totals[index]['topk'] += topk_count
                level_totals[index]['topk_valid'] += topk_valid
                level_totals[index]['topk_padding'] += topk_padding
                selected_logits = topk_logits[selected]
                score_valid.extend(selected_logits[local_valid].tolist())
                score_padding.extend(selected_logits[~local_valid].tolist())
            valid_pixels = int((pad_mask > 0.5).sum())
            total_pixels = int(pad_mask.size)
            row = {
                'image_id': image_id,
                'file_name': image_by_id[image_id]['file_name'],
                'gt_count': len(annotations_by_image.get(image_id, [])),
                'input_valid_pixels': valid_pixels,
                'input_padding_pixels': total_pixels - valid_pixels,
                'input_padding_ratio': (total_pixels - valid_pixels) / total_pixels,
                'pad_mask_min': float(pad_mask.min()),
                'pad_mask_max': float(pad_mask.max()),
                'pad_mask_mean': float(pad_mask.mean()),
                'topk_total': int(topk_indices.size),
                'topk_valid': int(topk_indices.size - selected_padding),
                'topk_padding': int(selected_padding),
                'topk_padding_ratio': selected_padding / float(topk_indices.size),
                'valid_topk_score_mean': float(np.mean([
                    score for score, index in zip(topk_logits, topk_indices)
                    if any(feature_masks[level][index - level_offsets[level]]
                           for level in range(len(feature_masks))
                           if level_offsets[level] <= index < level_offsets[level + 1])
                ])) if selected_padding < topk_indices.size else float('nan'),
                'padding_topk_score_mean': float(np.mean([
                    score for score, index in zip(topk_logits, topk_indices)
                    if any(not feature_masks[level][index - level_offsets[level]]
                           for level in range(len(feature_masks))
                           if level_offsets[level] <= index < level_offsets[level + 1])
                ])) if selected_padding else float('nan'),
            }
            for index in range(len(runtime_levels)):
                prefix = 'level{}_'.format(index)
                row[prefix + 'name'] = runtime_levels[index]['name']
                row[prefix + 'topk'] = selected_by_level[index]
                row[prefix + 'padding'] = selected_padding_by_level[index]
            rows.append(row)
            if (step_id + 1) % 25 == 0 or step_id + 1 == len(trainer.loader):
                print('Padding diagnostic: {}/{} images'.format(
                    step_id + 1, len(trainer.loader)), flush=True)
    if len(rows) != 400:
        raise RuntimeError('Expected 400 validation images, got {}'.format(len(rows)))
    if production_topk_checks != 10:
        raise RuntimeError('Expected 10 production TopK checks, got {}'.format(
            production_topk_checks))
    fields = list(rows[0].keys())
    write_csv(args.output_dir / 'topk_padding_per_image.csv', rows, fields)
    summary = {
        'images': len(rows),
        'input_padding_ratio': summary_stats([row['input_padding_ratio'] for row in rows]),
        'pad_mask_min': summary_stats([row['pad_mask_min'] for row in rows]),
        'pad_mask_max': summary_stats([row['pad_mask_max'] for row in rows]),
        'pad_mask_mean': summary_stats([row['pad_mask_mean'] for row in rows]),
        'topk_padding_count': summary_stats([row['topk_padding'] for row in rows]),
        'topk_padding_ratio': summary_stats([row['topk_padding_ratio'] for row in rows]),
        'levels': [{**runtime_levels[index], **level_totals[index]}
                   for index in range(len(runtime_levels))],
        'valid_topk_logits': summary_stats(score_valid),
        'padding_topk_logits': summary_stats(score_padding),
        'production_topk_equivalence_checks': production_topk_checks,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(args.output_dir / 'report.md', args, git_before, rows,
                 level_totals, runtime_levels, score_valid, score_padding,
                 environment)
    print('Padding diagnostic complete: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
