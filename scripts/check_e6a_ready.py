#!/usr/bin/env python3
"""Fail-closed readiness gate for the Sol E6a implementation."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path

from ppdet.core.workspace import create, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_sol_960_dual_o2m.yml'
TRAIN_JSON = ROOT / 'dataset/aic2026_qadepth/annotations_stratified_candidate/train.json'
VAL_JSON = ROOT / 'dataset/aic2026_qadepth/annotations_stratified_candidate/val.json'
EXPECTED_HASHES = {
    'ppdet/modeling/heads/ppyoloe_head.py':
        '2BDB1457A71C2280267CF87D73ECBB8FCEA5559806825048B90F918B03EC0035',
    'ppdet/modeling/assigners/atss_assigner.py':
        '0F725D402C42F518D463FC0EB6CD91BF2BFD86889B9830DD73F2331A4450368B',
    'ppdet/modeling/assigners/task_aligned_assigner.py':
        '4CED1D0A06EE48B29E0A5B5FE057320EAA962809078AE0076636BBEECF978C36',
}


def count_present_images(annotation_path, image_dir):
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    names = [entry['file_name'] for entry in payload['images']]
    return len(names), sum((image_dir / name).is_file() for name in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--acceptance-json',
        default='output/E6a_Sol_RGBIR_DAMSDet_960_dual_o2m/acceptance/acceptance.json')
    args = parser.parse_args()

    for relative, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper()
        if actual != expected:
            raise AssertionError('{} hash mismatch: {}'.format(relative, actual))

    train_presence = {}
    val_presence = {}
    train_total = val_total = 0
    for modality in ('visible', 'infrared', 'depth'):
        train_total, train_presence[modality] = count_present_images(
            TRAIN_JSON, ROOT / 'data/AIC2026_Train_2000' / modality)
        val_total, val_presence[modality] = count_present_images(
            VAL_JSON, ROOT / 'data/AIC2026_Train_2000' / modality)
    if (train_total, val_total) != (1600, 400):
        raise AssertionError('Expected 1600/400 split, got {}/{}'.format(
            train_total, val_total))
    if any(value != 1600 for value in train_presence.values()) or any(
            value != 400 for value in val_presence.values()):
        raise AssertionError('Modality file counts: train {}, val {}'.format(
            train_presence, val_presence))

    cfg = load_config(str(CONFIG))
    model = create(cfg.architecture)
    if model.depth_encoder is not None or model.depth_gating is not None:
        raise AssertionError('E6a must not enable Depth.')
    if model.transformer.num_queries != 300:
        raise AssertionError('E6a must retain 300 queries.')
    if model.transformer.num_decoder_layers != 6:
        raise AssertionError('E6a must retain six decoder layers.')
    if model.aux_o2m_head_vis.in_channels != [256, 256, 256]:
        raise AssertionError('VIS auxiliary channels are not [256, 256, 256].')
    if model.aux_o2m_head_ir.in_channels != [256, 256, 256]:
        raise AssertionError('IR auxiliary channels are not [256, 256, 256].')
    if model.aux_o2m_head_vis.static_assigner_epoch != 30:
        raise AssertionError('static_assigner_epoch must be 30.')
    if model.aux_o2m_head_ir.static_assigner_epoch != 30:
        raise AssertionError('static_assigner_epoch must be 30.')
    if model.aux_o2m_head_vis is model.aux_o2m_head_ir:
        raise AssertionError('VIS and IR auxiliary heads must be distinct Layers.')
    vis_parameter_ids = {
        id(parameter) for parameter in model.aux_o2m_head_vis.parameters()}
    ir_parameter_ids = {
        id(parameter) for parameter in model.aux_o2m_head_ir.parameters()}
    if not vis_parameter_ids or not ir_parameter_ids:
        raise AssertionError('An auxiliary head has no parameters.')
    if not vis_parameter_ids.isdisjoint(ir_parameter_ids):
        raise AssertionError('VIS and IR auxiliary parameters must not be shared.')

    acceptance_path = ROOT / args.acceptance_json
    if not acceptance_path.is_file():
        raise FileNotFoundError(
            'Dynamic acceptance report is missing: {}'.format(acceptance_path))
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    if not acceptance.get('all_passed', False):
        raise AssertionError('Dynamic acceptance did not pass completely.')
    required = {
        'e1_exact_equivalence', 'aux_disabled_regression', 'gt_pipeline',
        'assigner_transition', 'positive_counts', 'gradients',
        'tiny_overfit', 'smoke_100', 'eval_aux_calls',
        'checkpoint_resume', 'empty_gt'}
    missing = sorted(required - set(acceptance.get('checks', {})))
    if missing:
        raise AssertionError('Acceptance report is missing: {}'.format(missing))
    print('E6a readiness check passed')


if __name__ == '__main__':
    main()
