#!/usr/bin/env python3
"""Fail-fast verification of the formal E0 experiment contract."""

from __future__ import print_function

import argparse
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
    with open(path, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)

    assert cfg.architecture == 'DAMSDet'
    model_cfg = cfg['DAMSDet']
    assert model_cfg['backbone_vis'] == 'ResNet'
    assert model_cfg['backbone_ir'] == 'ResNet'
    assert 'depth_encoder' not in model_cfg
    assert 'depth_gating' not in model_cfg
    assert list(cfg.eval_size) == [640, 640]
    assert int(cfg.num_classes) == 12
    assert int(cfg.TrainReader['batch_size']) == 1
    assert int(cfg.seed) == 2026
    assert bool(cfg.classwise)

    eval_resize = _transform_config(
        cfg.EvalReader['sample_transforms'], 'Multi_Resize')
    assert list(eval_resize['target_size']) == [640, 640]
    assert not bool(eval_resize['keep_ratio'])

    train_anno = os.path.join(REPO_ROOT, cfg.TrainDataset['anno_path'])
    val_anno = os.path.join(REPO_ROOT, cfg.EvalDataset['anno_path'])
    train_coco = _read_coco(train_anno)
    val_coco = _read_coco(val_anno)
    assert len(train_coco['images']) == 1600
    assert len(val_coco['images']) == 400
    assert len(train_coco['categories']) == 12
    assert len(val_coco['categories']) == 12

    pretrain_path = cfg.pretrain_weights
    if not os.path.isabs(pretrain_path):
        pretrain_path = os.path.join(REPO_ROOT, pretrain_path)
    assert os.path.isfile(pretrain_path), pretrain_path

    print('E0 contract verified:')
    print('  architecture : official RGB+IR DAMSDet (no Depth branch)')
    print('  resolution   : train multi-scale around 640; eval 640x640')
    print('  split        : 1600 train / 400 val; 12 categories')
    print('  pretrained   : {} ({:.1f} MiB)'.format(
        os.path.basename(pretrain_path),
        os.path.getsize(pretrain_path) / (1024.0 * 1024.0)))
    print('  schedule     : {} epochs; best checkpoint by COCO mAP@50-95'.format(
        int(cfg.epoch)))
    print('  output       : {}'.format(cfg.save_dir))


if __name__ == '__main__':
    main()
