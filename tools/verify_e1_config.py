#!/usr/bin/env python3
"""Fail-fast verification of the E0 -> E1 experiment contract."""

from __future__ import print_function

import argparse
import copy
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from ppdet.core.workspace import load_config  # noqa: E402


def _transform_config(transforms, name):
    for item in transforms:
        if name in item:
            return item[name]
    raise AssertionError('{} is missing from reader transforms'.format(name))


def _read_coco(path):
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def _plain(value):
    """Detach a loaded config subtree from workspace's global config."""
    return copy.deepcopy(dict(value)) if isinstance(value, dict) else copy.deepcopy(value)


def _canonical(value):
    """Convert instantiated YAML objects to comparable plain structures."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, '__dict__'):
        return {
            '__class__': value.__class__.__name__,
            'fields': _canonical(vars(value)),
        }
    return value


def _load_snapshot(path):
    cfg = load_config(os.path.abspath(path))
    keys = [
        'architecture', 'DAMSDet', 'ResNet', 'HybridEncoder',
        'DAMS_DETR_Transformer', 'DINOHead', 'DETRPostProcess',
        'TrainReader', 'EvalReader', 'TestReader', 'TrainDataset',
        'EvalDataset', 'LearningRate', 'OptimizerBuilder', 'epoch',
        'snapshot_epoch', 'seed', 'num_classes', 'classwise', 'eval_size',
        'pretrain_weights', 'save_dir'
    ]
    return {key: _plain(cfg[key]) for key in keys if key in cfg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--e0',
        default='configs/damsdet/damsdet_r50vd_aic2026_official_rgbir.yml')
    parser.add_argument(
        '-c', '--config',
        default='configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml')
    parser.add_argument(
        '--skip-pretrained-check',
        action='store_true',
        help='Verify code/config before the large checkpoint is uploaded.')
    args = parser.parse_args()

    e0 = _load_snapshot(args.e0)
    e1 = _load_snapshot(args.config)

    assert e1['architecture'] == e0['architecture'] == 'DAMSDet'
    assert e1['DAMSDet'] == e0['DAMSDet']
    assert 'depth_encoder' not in e1['DAMSDet']
    assert 'depth_gating' not in e1['DAMSDet']
    for key in ('ResNet', 'HybridEncoder', 'DINOHead'):
        assert e1[key] == e0[key], '{} changed between E0 and E1'.format(key)

    e0_transformer = copy.deepcopy(e0['DAMS_DETR_Transformer'])
    e1_transformer = copy.deepcopy(e1['DAMS_DETR_Transformer'])
    assert e0_transformer == e1_transformer
    assert int(e1_transformer['num_queries']) == 300
    assert int(e1_transformer['num_decoder_layers']) == 6

    assert list(e0['eval_size']) == [640, 640]
    assert list(e1['eval_size']) == [960, 960]
    assert e1['TrainDataset'] == e0['TrainDataset']
    assert e1['EvalDataset'] == e0['EvalDataset']
    assert _canonical(e1['LearningRate']) == _canonical(e0['LearningRate'])
    assert _canonical(e1['OptimizerBuilder']) == _canonical(
        e0['OptimizerBuilder'])
    for key in ('epoch', 'snapshot_epoch', 'seed', 'num_classes',
                'classwise', 'pretrain_weights'):
        assert e1[key] == e0[key], '{} changed between E0 and E1'.format(key)

    e0_train = e0['TrainReader']
    e1_train = e1['TrainReader']
    assert e1_train['sample_transforms'] == e0_train['sample_transforms']
    for key in ('batch_size', 'shuffle', 'drop_last', 'collate_batch',
                'use_shared_memory'):
        assert e1_train[key] == e0_train[key]

    e0_resize = _transform_config(
        e0_train['batch_transforms'], 'Multi_BatchRandomResize')
    e1_resize = _transform_config(
        e1_train['batch_transforms'], 'Multi_BatchRandomResize')
    e0_sizes = list(e0_resize['target_size'])
    e1_sizes = list(e1_resize['target_size'])
    assert e1_sizes == [int(round(size * 1.5)) for size in e0_sizes]
    assert not bool(e0_resize['keep_ratio'])
    assert bool(e1_resize['keep_ratio'])
    assert bool(e1_resize['pad_to_target'])
    assert int(e1_resize['pad_mode']) == 0
    assert bool(e1_resize['return_pad_mask'])
    assert e1_resize['random_size'] == e0_resize['random_size']
    assert e1_resize['random_interp'] == e0_resize['random_interp']
    assert e1_train['batch_transforms'][1:] == e0_train['batch_transforms'][1:]

    eval_resize = _transform_config(
        e1['EvalReader']['sample_transforms'], 'Multi_Resize')
    eval_pad = _transform_config(
        e1['EvalReader']['sample_transforms'], 'Multi_Pad_IRVIS')
    assert list(eval_resize['target_size']) == [960, 960]
    assert bool(eval_resize['keep_ratio'])
    assert list(eval_pad['size']) == [960, 960]
    assert int(eval_pad['pad_mode']) == 0
    assert bool(eval_pad['return_pad_mask'])
    assert e1['DETRPostProcess']['bbox_decode_type'] == 'pad'
    assert int(e1['DETRPostProcess']['num_top_queries']) == 300

    train_coco = _read_coco(e1['TrainDataset']['anno_path'])
    val_coco = _read_coco(e1['EvalDataset']['anno_path'])
    assert len(train_coco['images']) == 1600
    assert len(val_coco['images']) == 400
    assert len(train_coco['categories']) == 12
    assert len(val_coco['categories']) == 12

    pretrain_path = e1['pretrain_weights']
    if not os.path.isabs(pretrain_path):
        pretrain_path = os.path.join(REPO_ROOT, pretrain_path)
    if not args.skip_pretrained_check:
        assert os.path.isfile(pretrain_path), pretrain_path

    print('E1 contract verified:')
    print('  architecture : official RGB+IR DAMSDet (no Depth branch)')
    print('  train scales : {} (exactly E0 x 1.5)'.format(e1_sizes))
    print('  eval geometry: 1920x1080 -> 960x540 -> 960x960 bottom pad')
    print('  bbox decode  : padded canvas -> original image coordinates')
    print('  split        : 1600 train / 400 val; 12 categories')
    print('  schedule     : unchanged, {} epochs'.format(int(e1['epoch'])))
    print('  output       : {}'.format(e1['save_dir']))
    if args.skip_pretrained_check:
        print('  pretrained   : asset check skipped (configuration only)')
    print('NOTICE: E0 used stretched square resize; requested E1 also changes')
    print('        geometry to letterbox, so it is not a resolution-only causal test.')


if __name__ == '__main__':
    main()
