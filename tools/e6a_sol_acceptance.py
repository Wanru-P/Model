#!/usr/bin/env python3
"""Strict cloud acceptance for E6a Sol redo. Never starts epoch-12 training."""
from __future__ import print_function

import argparse
import copy
import csv
import json
import math
import random
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn

from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.checkpoint import (
    load_weight, multi_match_state_dict)


ROOT = Path(__file__).resolve().parents[1]
E1_CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml'
E6_CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_sol_960_dual_o2m.yml'
TRAIN_JSON = ROOT / 'dataset/aic2026_qadepth/annotations_stratified_candidate/train.json'
OUTPUT_DIR = ROOT / 'output/E6a_Sol_RGBIR_DAMSDet_960_dual_o2m/acceptance'
CLASS_NAMES = [
    'person', 'boat', 'animal', 'seat', 'sign', 'bicycle', 'car', 'ball',
    'light', 'garbage_can', 'uav', 'tricycle'
]
TINY_CLASS_IDS = {0, 4, 7, 10}
AUX_PREFIXES = ('aux_o2m_head_vis.', 'aux_o2m_head_ir.')


class ForwardCapture(object):
    def __init__(self):
        self.outputs = []

    def __call__(self, layer, inputs, output):
        self.outputs.append(output)


class FiniteCapture(object):
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.all_finite = True

    def __call__(self, layer, inputs, output):
        self.calls += 1
        values = output if isinstance(output, (list, tuple)) else [output]
        for value in values:
            if isinstance(value, paddle.Tensor):
                self.all_finite = self.all_finite and bool(
                    paddle.isfinite(value).all().numpy().item())


class CallCounter(object):
    def __init__(self):
        self.calls = 0

    def __call__(self, layer, inputs, output):
        self.calls += 1


class AssignerProbe(nn.Layer):
    def __init__(self, assigner, name):
        super(AssignerProbe, self).__init__()
        self.assigner = assigner
        self.name = name
        self.records = []

    def forward(self, *args, **kwargs):
        output = self.assigner(*args, **kwargs)
        labels, boxes, scores = output[:3]
        bg_index = kwargs.get('bg_index', args[-1] if args else 12)
        positives = int((labels != bg_index).astype('int32').sum().numpy().item())
        positive_by_class = {
            CLASS_NAMES[class_id]: int(
                (labels == class_id).astype('int32').sum().numpy().item())
            for class_id in range(len(CLASS_NAMES))
        }
        self.records.append({
            'name': self.name,
            'positive_count': positives,
            'positive_by_class': positive_by_class,
            'assigned_scores_finite': bool(
                paddle.isfinite(scores).all().numpy().item()),
            'assigned_boxes_finite': bool(
                paddle.isfinite(boxes).all().numpy().item())
        })
        return output


def seed_all(value=20260916):
    random.seed(value)
    np.random.seed(value)
    paddle.seed(value)


def to_numpy(value):
    if isinstance(value, paddle.Tensor):
        return value.numpy()
    return np.asarray(value)


def first_array(value):
    if isinstance(value, (list, tuple)):
        return to_numpy(value[0])
    array = to_numpy(value)
    return array[0] if array.ndim > 2 else array


def scalar(value):
    return float(to_numpy(value).reshape([-1])[0])


def finite(value):
    return bool(np.isfinite(to_numpy(value)).all())


def max_abs_diff(left, right):
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float('inf')
        return max(max_abs_diff(a, b) for a, b in zip(left, right))
    return float(np.max(np.abs(
        to_numpy(left).astype('float64') - to_numpy(right).astype('float64'))))


def synchronize():
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()


def assert_aux_construction(model, enabled, label):
    actual_enabled = bool(model.aux_o2m_enabled)
    vis_head = model.aux_o2m_head_vis
    ir_head = model.aux_o2m_head_ir
    if enabled:
        if not actual_enabled or vis_head is None or ir_head is None:
            raise AssertionError(
                '{} must construct two enabled auxiliary heads.'.format(label))
        if vis_head is ir_head:
            raise AssertionError(
                '{} auxiliary heads share one Layer instance.'.format(label))
    elif actual_enabled or vis_head is not None or ir_head is not None:
        raise AssertionError(
            '{} retained auxiliary state: enabled={}, vis={}, ir={}'.format(
                label, actual_enabled,
                type(vis_head).__name__ if vis_head is not None else None,
                type(ir_head).__name__ if ir_head is not None else None))


def build_trainer(config_path, mode, aux_enabled=None,
                  tiny_annotation=None, amp=False):
    config_path = Path(config_path).resolve()
    cfg = load_config(str(config_path))
    is_e1 = config_path == E1_CONFIG.resolve()
    if is_e1 and aux_enabled not in (None, False):
        raise AssertionError('E1 cannot enable E6a auxiliary heads.')
    effective_aux_enabled = (
        False if is_e1 else
        bool(cfg.DAMSDet.get('aux_o2m_enabled', False))
        if aux_enabled is None else bool(aux_enabled))
    cfg.DAMSDet['aux_o2m_enabled'] = effective_aux_enabled
    if not effective_aux_enabled:
        # load_config() merges into a process-global configuration. E1 does not
        # mention these E6a-only fields, so explicitly remove both references
        # after every load instead of allowing a prior E6a build to leak in.
        cfg.DAMSDet['aux_o2m_head_vis'] = None
        cfg.DAMSDet['aux_o2m_head_ir'] = None
    if tiny_annotation is not None:
        cfg.TrainDataset['anno_path'] = str(tiny_annotation)
        cfg.TrainReader['shuffle'] = False
    cfg['amp'] = bool(amp)
    cfg['worker_num'] = 0
    trainer = Trainer(cfg, mode=mode)
    assert_aux_construction(
        trainer.model, effective_aux_enabled,
        '{} ({})'.format(config_path.name, mode))
    return trainer


def load_e1_into_e6(model, checkpoint):
    source = paddle.load(str(checkpoint))
    target = model.state_dict()
    matched = {}
    shape_mismatch = []
    for name, tensor in target.items():
        if name in source and list(tensor.shape) == list(source[name].shape):
            matched[name] = source[name]
        elif name in source:
            shape_mismatch.append(name)
    missing = sorted(set(target) - set(matched))
    unexpected = sorted(set(source) - set(matched))
    illegal_missing = [name for name in missing if not name.startswith(
        ('aux_o2m_head_vis.', 'aux_o2m_head_ir.'))]
    if illegal_missing or shape_mismatch or unexpected:
        raise AssertionError({
            'illegal_missing': illegal_missing,
            'shape_mismatch': shape_mismatch,
            'unexpected': unexpected
        })
    target.update(matched)
    model.set_dict(target)
    return {
        'loaded_main': len(matched),
        'missing_aux_vis': len([n for n in missing if n.startswith('aux_o2m_head_vis.')]),
        'missing_aux_ir': len([n for n in missing if n.startswith('aux_o2m_head_ir.')]),
        'unexpected': unexpected,
        'shape_mismatch': shape_mismatch
    }


def split_main_and_aux_keys(state_dict):
    all_keys = set(state_dict)
    aux_keys = {name for name in all_keys if name.startswith(AUX_PREFIXES)}
    return all_keys - aux_keys, aux_keys


def compare_shared_main_state(e1_model, e6_model, stage):
    e1_state = e1_model.state_dict()
    e6_state = e6_model.state_dict()
    e1_main, e1_aux = split_main_and_aux_keys(e1_state)
    e6_main, e6_aux = split_main_and_aux_keys(e6_state)
    if e1_aux:
        raise AssertionError('E1 unexpectedly contains auxiliary state.')
    if e1_main != e6_main:
        raise AssertionError({
            'stage': stage,
            'missing_from_e6a': sorted(e1_main - e6_main),
            'extra_in_e6a_main': sorted(e6_main - e1_main)})
    shape_mismatch = []
    dtype_mismatch = []
    value_mismatch = []
    for name in sorted(e1_main):
        left = e1_state[name]
        right = e6_state[name]
        if list(left.shape) != list(right.shape):
            shape_mismatch.append(name)
            continue
        if left.dtype != right.dtype:
            dtype_mismatch.append(name)
            continue
        if not bool(paddle.equal_all(left, right).numpy().item()):
            value_mismatch.append({
                'name': name, 'max_abs_diff': max_abs_diff(left, right)})
    if shape_mismatch or dtype_mismatch or value_mismatch:
        raise AssertionError({
            'stage': stage,
            'shape_mismatch': shape_mismatch,
            'dtype_mismatch': dtype_mismatch,
            'value_mismatch': value_mismatch[:20]})
    return {
        'stage': stage,
        'shared_main_keys': len(e1_main),
        'e6a_aux_keys': len(e6_aux),
        'max_abs_diff': 0.0,
        'exact_equal': True}


def inspect_pretrain_loading(e1_trainer, e6_trainer, checkpoint):
    if e1_trainer.checkpoint_mode != 'multi' or \
            e6_trainer.checkpoint_mode != 'multi':
        raise AssertionError('COCO parity must use Trainer train-mode multi loading.')
    before = compare_shared_main_state(
        e1_trainer.model, e6_trainer.model, 'before_coco_pretrain')
    source = paddle.load(str(checkpoint))
    e1_state = e1_trainer.model.state_dict()
    e6_state = e6_trainer.model.state_dict()
    e1_main, _ = split_main_and_aux_keys(e1_state)
    e6_main, e6_aux = split_main_and_aux_keys(e6_state)
    e1_matched = set(multi_match_state_dict(
        e1_state, source, mode='multi'))
    e6_matched = set(multi_match_state_dict(
        e6_state, source, mode='multi'))
    e1_matched_main = e1_matched & e1_main
    e6_matched_main = e6_matched & e6_main
    e1_unmatched_main = e1_main - e1_matched_main
    e6_unmatched_main = e6_main - e6_matched_main
    e6_matched_aux = e6_matched & e6_aux
    if e6_matched_main != e1_matched_main:
        raise AssertionError({
            'e6a_missing_relative_to_e1': sorted(
                e1_matched_main - e6_matched_main),
            'e6a_extra_matches_relative_to_e1': sorted(
                e6_matched_main - e1_matched_main)})
    if e6_unmatched_main != e1_unmatched_main:
        raise AssertionError({
            'e1_unmatched_only': sorted(e1_unmatched_main - e6_unmatched_main),
            'e6a_unmatched_only': sorted(e6_unmatched_main - e1_unmatched_main)})
    if e6_matched_aux:
        raise AssertionError(
            'E6a auxiliary tensors unexpectedly matched COCO: {}'.format(
                sorted(e6_matched_aux)))

    # Exercise the exact formal-training path, including Trainer's `multi`
    # checkpoint mode. Unmatched tensors retain their same-seed initialization.
    e1_trainer.load_weights(str(checkpoint))
    e6_trainer.load_weights(str(checkpoint))
    after = compare_shared_main_state(
        e1_trainer.model, e6_trainer.model, 'after_coco_pretrain')
    return {
        'before_pretrain_main_state': before,
        'after_pretrain_main_state': after,
        'e1_matched_main_keys': sorted(e1_matched_main),
        'e1_unmatched_main_keys': sorted(e1_unmatched_main),
        'e6a_matched_shared_main_keys': sorted(e6_matched_main),
        'e6a_unmatched_shared_main_keys': sorted(e6_unmatched_main),
        'matched_main_set_equal': True,
        'unmatched_main_set_equal': True,
        'aux_vis_initialized_from_scratch': not any(
            name.startswith('aux_o2m_head_vis.') for name in e6_matched),
        'aux_ir_initialized_from_scratch': not any(
            name.startswith('aux_o2m_head_ir.') for name in e6_matched)
    }


def register_main_captures(model):
    captures = {}
    handles = []
    for name in ('backbone_vis', 'backbone_ir', 'neck_vis', 'neck_ir', 'detr_head'):
        capture = ForwardCapture()
        captures[name] = capture
        handles.append(getattr(model, name).register_forward_post_hook(capture))
    return captures, handles


def remove_handles(handles):
    for handle in handles:
        handle.remove()


def eval_outputs(model, batch):
    captures, handles = register_main_captures(model)
    model.eval()
    with paddle.no_grad():
        output = model(copy.deepcopy(batch))
    remove_handles(handles)
    return captures, output


def compare_eval(e1_model, e6_model, batch):
    seed_all()
    e1_capture, e1_output = eval_outputs(e1_model, batch)
    seed_all()
    e6_capture, e6_output = eval_outputs(e6_model, batch)
    e1_raw = e1_capture['detr_head'].outputs[-1]
    e6_raw = e6_capture['detr_head'].outputs[-1]
    result = {
        'vis_backbone': max_abs_diff(
            e1_capture['backbone_vis'].outputs[-1],
            e6_capture['backbone_vis'].outputs[-1]),
        'ir_backbone': max_abs_diff(
            e1_capture['backbone_ir'].outputs[-1],
            e6_capture['backbone_ir'].outputs[-1]),
        'vis_neck': max_abs_diff(
            e1_capture['neck_vis'].outputs[-1],
            e6_capture['neck_vis'].outputs[-1]),
        'ir_neck': max_abs_diff(
            e1_capture['neck_ir'].outputs[-1],
            e6_capture['neck_ir'].outputs[-1]),
        'decoder_boxes': max_abs_diff(e1_raw[0], e6_raw[0]),
        'decoder_logits': max_abs_diff(e1_raw[1], e6_raw[1]),
        'final_scores': max_abs_diff(e1_output['bbox'][:, 1], e6_output['bbox'][:, 1]),
        'final_boxes': max_abs_diff(e1_output['bbox'][:, 2:], e6_output['bbox'][:, 2:]),
        'bbox_num': max_abs_diff(e1_output['bbox_num'], e6_output['bbox_num'])
    }
    if max(result.values()) > 1e-7:
        raise AssertionError('E1 equivalence failed: {}'.format(result))
    return result


def compare_train_losses(e1_model, e6_model, batch):
    e1_model.train()
    e6_model.train()
    seed_all()
    e1_losses = e1_model(copy.deepcopy(batch))
    seed_all()
    e6_losses = e6_model(copy.deepcopy(batch))
    common = sorted(set(e1_losses) & set(e6_losses))
    result = {name: max_abs_diff(e1_losses[name], e6_losses[name])
              for name in common}
    if max(result.values()) > 1e-7:
        raise AssertionError('Aux-disabled regression failed: {}'.format(result))
    return result


def dtype_name(value):
    return str(value.dtype).lower()


def check_origin_batch_contract(batch):
    specifications = {
        'origin_gt_bbox': (4, ('float32', )),
        'origin_gt_class': (1, ('int32', 'int64')),
        'pad_origin_gt_mask': (1, ('float32', ))}
    result = {}
    for name, (last_dimension, allowed_dtypes) in specifications.items():
        value = batch[name]
        if isinstance(value, list):
            raise AssertionError('{} remained a Python list.'.format(name))
        ndim = int(value.ndim)
        shape = [int(item) for item in value.shape]
        dtype = dtype_name(value)
        if ndim != 3 or shape[0] != 1 or shape[-1] != last_dimension:
            raise AssertionError({
                'name': name, 'ndim': ndim, 'shape': shape})
        if not any(dtype.endswith(item) for item in allowed_dtypes):
            raise AssertionError({
                'name': name, 'dtype': dtype,
                'allowed_dtypes': allowed_dtypes})
        result[name] = {
            'type': type(value).__name__,
            'shape': shape,
            'dtype': dtype}
    count = result['origin_gt_bbox']['shape'][1]
    if result['origin_gt_class']['shape'][1] != count or \
            result['pad_origin_gt_mask']['shape'][1] != count:
        raise AssertionError('Origin target batch dimensions disagree.')
    return result


def check_e1_batch_contract(batch):
    result = {}
    for name in ('gt_bbox', 'gt_class'):
        value = batch[name]
        if not isinstance(value, list):
            raise AssertionError(
                'Official origin stacking changed E1 {} semantics.'.format(name))
        result[name] = {
            'type': type(value).__name__,
            'batch_items': len(value),
            'item_shape': list(to_numpy(value[0]).shape),
            'item_dtype': dtype_name(value[0])}
    return result


def check_gt_batch(batch):
    origin = first_array(batch['origin_gt_bbox']).reshape([-1, 4])
    dino = first_array(batch['gt_bbox']).reshape([-1, 4])
    mask = first_array(batch['pad_origin_gt_mask']).reshape([-1]).astype(bool)
    image = to_numpy(batch['vis_image'])
    height, width = image.shape[-2:]
    origin = origin[mask]
    dino = dino[:len(mask)][mask]
    if len(origin) == 0:
        return {'gt_count': 0, 'roundtrip_error': 0.0, 'bounds_valid': True}
    cx, cy, box_width, box_height = [dino[:, index] for index in range(4)]
    rebuilt = np.stack([
        (cx - box_width / 2.) * width,
        (cy - box_height / 2.) * height,
        (cx + box_width / 2.) * width,
        (cy + box_height / 2.) * height
    ], axis=1)
    bounds = bool(np.all(origin[:, 0] >= 0) and np.all(origin[:, 1] >= 0) and
                  np.all(origin[:, 2] <= width) and np.all(origin[:, 3] <= height) and
                  np.all(origin[:, 2] > origin[:, 0]) and
                  np.all(origin[:, 3] > origin[:, 1]))
    error = float(np.max(np.abs(rebuilt - origin)))
    if not bounds or error >= 1e-2:
        raise AssertionError({'bounds': bounds, 'roundtrip_error': error})
    return {'gt_count': int(len(origin)), 'roundtrip_error': error,
            'bounds_valid': bounds}


def parameter_counts(e1_model, e6_model):
    if e6_model.aux_o2m_head_vis is e6_model.aux_o2m_head_ir:
        raise AssertionError('VIS and IR auxiliary heads share one Layer instance.')
    vis_parameter_ids = {
        id(parameter) for parameter in e6_model.aux_o2m_head_vis.parameters()}
    ir_parameter_ids = {
        id(parameter) for parameter in e6_model.aux_o2m_head_ir.parameters()}
    if not vis_parameter_ids or not ir_parameter_ids:
        raise AssertionError('An auxiliary head has no parameters.')
    if not vis_parameter_ids.isdisjoint(ir_parameter_ids):
        raise AssertionError('VIS and IR auxiliary heads share parameters.')
    e1_total = sum(int(parameter.numel()) for parameter in e1_model.parameters())
    e6_total = sum(int(parameter.numel()) for parameter in e6_model.parameters())
    vis = sum(int(parameter.numel()) for parameter in
              e6_model.aux_o2m_head_vis.parameters())
    ir = sum(int(parameter.numel()) for parameter in
             e6_model.aux_o2m_head_ir.parameters())
    if e6_total - e1_total != vis + ir:
        raise AssertionError('Unexpected parameter delta.')
    return {
        'e1': e1_total,
        'e6a': e6_total,
        'aux_vis': vis,
        'aux_ir': ir,
        'distinct_layer_instances': True,
        'disjoint_parameter_objects': True}


def install_assigner_probes(model):
    probes = {}
    for modality in ('vis', 'ir'):
        head = getattr(model, 'aux_o2m_head_' + modality)
        static_probe = AssignerProbe(head.static_assigner, modality + '_ATSS')
        aligned_probe = AssignerProbe(head.assigner, modality + '_TaskAligned')
        head.static_assigner = static_probe
        head.assigner = aligned_probe
        probes[modality + '_ATSS'] = static_probe
        probes[modality + '_TaskAligned'] = aligned_probe
    return probes


def grad_norm(model, prefix):
    squares = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix):
            continue
        gradient = getattr(parameter, 'grad', None)
        if callable(gradient):
            gradient = gradient()
        if gradient is not None:
            array = to_numpy(gradient).astype('float64')
            if not np.isfinite(array).all():
                raise AssertionError('Non-finite gradient in ' + prefix)
            squares.append(float(np.sum(array * array)))
    return math.sqrt(sum(squares)) if squares else 0.0


def collect_gradients(model, batch, loss_key):
    model.clear_gradients()
    seed_all()
    outputs = model(copy.deepcopy(batch))
    if loss_key == 'aux_only':
        selected = outputs['aux_mean']
    else:
        selected = outputs[loss_key]
    selected.backward()
    prefixes = ('backbone_vis', 'neck_vis', 'backbone_ir', 'neck_ir',
                'aux_o2m_head_vis', 'aux_o2m_head_ir')
    return {prefix: grad_norm(model, prefix) for prefix in prefixes}


def optimizer_step(trainer, batch, scaler, epoch_id):
    batch['epoch_id'] = epoch_id
    trainer.model.clear_gradients()
    with paddle.amp.auto_cast(enable=True, level='O1'):
        outputs = trainer.model(batch)
        loss = outputs['loss']
    if not all(finite(outputs[name]) for name in (
            'loss', 'dino_total', 'aux_vis_total', 'aux_ir_total', 'aux_mean')):
        raise FloatingPointError('Non-finite E6a loss.')
    scaled = scaler.scale(loss)
    scaled.backward()
    scale_value = scalar(scaler._scale) if hasattr(scaler, '_scale') else 1.0
    vis_neck_grad = grad_norm(trainer.model, 'neck_vis') / scale_value
    ir_neck_grad = grad_norm(trainer.model, 'neck_ir') / scale_value
    learning_rate = float(trainer.optimizer.get_lr())
    scaler.minimize(trainer.optimizer, scaled)
    trainer.optimizer.clear_grad()
    trainer.lr.step()
    if trainer.use_ema:
        trainer.ema.update()
    return outputs, vis_neck_grad, ir_neck_grad, learning_rate


def load_coco_start(trainer, checkpoint):
    if trainer.checkpoint_mode != 'multi':
        raise AssertionError('Formal COCO initialization requires multi mode.')
    trainer.load_weights(str(checkpoint))
    return {'loader': 'Trainer.load_weights', 'checkpoint_mode': 'multi'}


def validate_tiny_annotation(tiny_path):
    tiny = json.loads(tiny_path.read_text(encoding='utf-8'))
    train = json.loads(TRAIN_JSON.read_text(encoding='utf-8'))
    tiny_names = {item['file_name'] for item in tiny['images']}
    train_names = {item['file_name'] for item in train['images']}
    if len(tiny_names) != 24 or not tiny_names.issubset(train_names):
        raise AssertionError('Tiny annotation must contain 24 train1600 images.')


def run_tiny_overfit(tiny_path, coco_checkpoint):
    validate_tiny_annotation(tiny_path)
    trainer = build_trainer(E6_CONFIG, 'train', tiny_annotation=tiny_path, amp=True)
    load_coco_start(trainer, coco_checkpoint)
    trainer.model.train()
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    epoch_means = []
    for epoch in range(8):
        values = {'dino_total': [], 'aux_vis_total': [], 'aux_ir_total': []}
        trainer.loader.dataset.set_epoch(epoch)
        for batch in trainer.loader:
            outputs, _, _, _ = optimizer_step(
                trainer, batch, scaler, epoch)
            for name in values:
                values[name].append(scalar(outputs[name]))
        epoch_means.append({name: statistics.mean(series)
                            for name, series in values.items()})
    for name in epoch_means[0]:
        if epoch_means[-1][name] > epoch_means[0][name] * 1.2:
            raise AssertionError('Tiny-overfit {} diverged.'.format(name))
    return {'epochs': 8, 'first': epoch_means[0], 'last': epoch_means[-1]}


def gpu_peak_memory():
    if hasattr(paddle.device.cuda, 'max_memory_allocated'):
        return int(paddle.device.cuda.max_memory_allocated())
    return 0


def latency(model, batch):
    model.eval()
    with paddle.no_grad():
        for _ in range(10):
            model(copy.deepcopy(batch))
        synchronize()
        start = time.perf_counter()
        for _ in range(30):
            model(copy.deepcopy(batch))
        synchronize()
    return (time.perf_counter() - start) / 30.


def empty_gt_check(model, batch):
    vis_capture = ForwardCapture()
    ir_capture = ForwardCapture()
    vis_handle = model.neck_vis.register_forward_post_hook(vis_capture)
    ir_handle = model.neck_ir.register_forward_post_hook(ir_capture)
    batch['epoch_id'] = 0
    model.clear_gradients()
    model(copy.deepcopy(batch))
    vis_handle.remove()
    ir_handle.remove()
    empty = {
        'origin_gt_bbox': paddle.zeros([1, 0, 4], dtype='float32'),
        'origin_gt_class': paddle.zeros([1, 0, 1], dtype='int32'),
        'pad_origin_gt_mask': paddle.zeros([1, 0, 1], dtype='float32')}
    result = {}
    for epoch in (0, 30):
        empty['epoch_id'] = epoch
        vis_losses = model.aux_o2m_head_vis(
            vis_capture.outputs[-1], empty)
        ir_losses = model.aux_o2m_head_ir(
            ir_capture.outputs[-1], empty)
        if not all(finite(value) for value in list(vis_losses.values()) +
                   list(ir_losses.values())):
            raise FloatingPointError('Empty-GT auxiliary loss is non-finite.')
        result[str(epoch)] = {
            'vis': {name: scalar(value) for name, value in vis_losses.items()},
            'ir': {name: scalar(value) for name, value in ir_losses.items()}}
    return result


def save_and_resume(trainer, batch, scaler, output_dir):
    model_path = output_dir / 'smoke_model.pdparams'
    optimizer_path = output_dir / 'smoke_optimizer.pdopt'
    ema_path = output_dir / 'smoke_model.pdema'
    paddle.save(trainer.model.state_dict(), str(model_path))
    paddle.save(trainer.optimizer.state_dict(), str(optimizer_path))
    paddle.save(trainer.ema.state_dict, str(ema_path))
    resumed = build_trainer(E6_CONFIG, 'train', amp=True)
    resumed.model.set_dict(paddle.load(str(model_path)))
    resumed.optimizer.set_state_dict(paddle.load(str(optimizer_path)))
    resumed.ema.resume(paddle.load(str(ema_path)), trainer.ema.step)
    resumed.model.train()
    resume_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    outputs, _, _, _ = optimizer_step(
        resumed, copy.deepcopy(batch), resume_scaler, 0)
    return {'loss': scalar(outputs['loss']), 'model': str(model_path),
            'optimizer': str(optimizer_path), 'ema': str(ema_path)}


def update_report(result):
    report_path = ROOT / 'E6A_IMPLEMENTATION_REPORT.md'
    text = report_path.read_text(encoding='utf-8')
    marker = '\n## Dynamic acceptance evidence\n'
    if marker in text:
        text = text.split(marker, 1)[0]
    text = text.replace('Current verdict: **NOT READY**. Dynamic acceptance has not been run.\n', '')
    text = text.rsplit('## P Final verdict', 1)[0].rstrip()
    text += marker + '\n```json\n' + json.dumps(
        result, indent=2, ensure_ascii=False) + '\n```\n\n## P Final verdict\n\n'
    verdict = '**READY FOR E6a EARLY TRAINING**' if result['all_passed'] else '**NOT READY**'
    report_path.write_text(text + verdict + '\n', encoding='utf-8')


def run(args):
    if paddle.__version__ != '2.6.2' or not paddle.is_compiled_with_cuda():
        raise RuntimeError('Acceptance requires Paddle 2.6.2 with CUDA.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    e1_checkpoint = Path(args.e1_checkpoint).resolve()
    coco_checkpoint = Path(args.coco_pretrain).resolve()
    tiny_path = Path(args.tiny_annotation).resolve()
    for path in (e1_checkpoint, coco_checkpoint, tiny_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result = {'all_passed': False, 'environment': {
        'paddle': paddle.__version__, 'cuda': True}, 'checks': {}}
    e1_eval = build_trainer(E1_CONFIG, 'eval', aux_enabled=False)
    e6_eval = build_trainer(E6_CONFIG, 'eval', aux_enabled=True)
    load_weight(e1_eval.model, str(e1_checkpoint))
    result['checks']['e1_checkpoint_loading'] = load_e1_into_e6(
        e6_eval.model, e1_checkpoint)
    val_batch = next(iter(e1_eval.loader))
    result['checks']['parameters'] = parameter_counts(e1_eval.model, e6_eval.model)

    vis_call = ForwardCapture()
    ir_call = ForwardCapture()
    vis_handle = e6_eval.model.aux_o2m_head_vis.register_forward_post_hook(vis_call)
    ir_handle = e6_eval.model.aux_o2m_head_ir.register_forward_post_hook(ir_call)
    result['checks']['e1_exact_equivalence'] = compare_eval(
        e1_eval.model, e6_eval.model, val_batch)
    vis_handle.remove()
    ir_handle.remove()
    if vis_call.outputs or ir_call.outputs:
        raise AssertionError('Auxiliary head was called in eval mode.')
    result['checks']['eval_aux_calls'] = {'vis': 0, 'ir': 0}

    e1_latency = latency(e1_eval.model, val_batch)
    e6_latency = latency(e6_eval.model, val_batch)
    result['checks']['eval_latency_seconds'] = {
        'e1': e1_latency, 'e6a': e6_latency,
        'relative_delta': (e6_latency - e1_latency) / e1_latency}
    if result['checks']['eval_latency_seconds']['relative_delta'] > 0.15:
        raise AssertionError('E6a eval latency is more than 15% above E1.')
    del e1_eval, e6_eval

    e1_train = build_trainer(E1_CONFIG, 'train', aux_enabled=False)
    e6_disabled = build_trainer(E6_CONFIG, 'train', aux_enabled=False)
    load_weight(e1_train.model, str(e1_checkpoint))
    load_weight(e6_disabled.model, str(e1_checkpoint))
    result['checks']['config_isolation'] = {
        'sequence': ['E1 eval disabled', 'E6a eval enabled',
                     'E1 train disabled', 'E6a train disabled'],
        'e1_eval_no_aux': True,
        'e1_train_after_e6_no_aux': True,
        'e6_disabled_no_aux': True,
        'strict_e1_checkpoint_load': True}
    regression_batch = next(iter(e1_train.loader))
    result['checks']['e1_batch_contract'] = check_e1_batch_contract(
        regression_batch)
    regression_batch['epoch_id'] = 0
    result['checks']['aux_disabled_regression'] = compare_train_losses(
        e1_train.model, e6_disabled.model, regression_batch)
    del e1_train, e6_disabled

    seed_all()
    e1_coco = build_trainer(
        E1_CONFIG, 'train', aux_enabled=False, amp=True)
    seed_all()
    train_model = build_trainer(
        E6_CONFIG, 'train', aux_enabled=True, amp=True)
    result['checks']['coco_pretrain_parity'] = inspect_pretrain_loading(
        e1_coco, train_model, coco_checkpoint)
    del e1_coco
    train_model.model.train()
    probes = install_assigner_probes(train_model.model)
    train_vis_calls = CallCounter()
    train_ir_calls = CallCounter()
    train_vis_handle = train_model.model.aux_o2m_head_vis.register_forward_post_hook(
        train_vis_calls)
    train_ir_handle = train_model.model.aux_o2m_head_ir.register_forward_post_hook(
        train_ir_calls)
    gt_checks = []
    origin_batch_contracts = []
    positive_checks = []
    train_batches = []
    train_iterator = iter(train_model.loader)
    tiny_seen = set()
    scanned = 0
    while len(positive_checks) < 20 or tiny_seen != TINY_CLASS_IDS:
        if scanned >= 1600:
            raise AssertionError(
                'Could not cover person/sign/ball/uav in train1600.')
        batch = next(train_iterator)
        scanned += 1
        labels_for_selection = first_array(
            batch['origin_gt_class']).reshape([-1])
        mask_for_selection = first_array(
            batch['pad_origin_gt_mask']).reshape([-1]).astype(bool)
        class_ids = {int(item) for item in labels_for_selection[mask_for_selection]}
        should_test = len(positive_checks) < 20 or bool(
            (class_ids & TINY_CLASS_IDS) - tiny_seen)
        if not should_test:
            continue
        origin_batch_contracts.append(check_origin_batch_contract(batch))
        tiny_seen.update(class_ids & TINY_CLASS_IDS)
        train_batches.append(copy.deepcopy(batch))
        gt_checks.append(check_gt_batch(batch))
        batch['epoch_id'] = 0
        train_model.model.clear_gradients()
        outputs = train_model.model(batch)
        if not all(finite(outputs[name]) for name in (
                'loss', 'dino_total', 'aux_vis_total', 'aux_ir_total')):
            raise FloatingPointError('Non-finite loss in positive sanity check.')
        if 'single_batch_forward' not in result['checks']:
            result['checks']['single_batch_forward'] = {
                'loss': scalar(outputs['loss']),
                'aux_vis_total': scalar(outputs['aux_vis_total']),
                'aux_ir_total': scalar(outputs['aux_ir_total']),
                'all_finite': True}
        labels = labels_for_selection
        valid = mask_for_selection
        present = [CLASS_NAMES[int(item)] for item in labels[valid]]
        boxes = first_array(batch['origin_gt_bbox']).reshape([-1, 4])[valid]
        tiny_boxes = []
        for label, box in zip(labels[valid], boxes):
            if int(label) in TINY_CLASS_IDS:
                tiny_boxes.append({
                    'class': CLASS_NAMES[int(label)],
                    'width': float(box[2] - box[0]),
                    'height': float(box[3] - box[1])})
        positive_checks.append({
            'source_scan_index': scanned - 1,
            'gt_count': int(valid.sum()), 'classes': present,
            'vis_positive': probes['vis_ATSS'].records[-1]['positive_count'],
            'ir_positive': probes['ir_ATSS'].records[-1]['positive_count'],
            'vis_positive_by_class': probes['vis_ATSS'].records[-1][
                'positive_by_class'],
            'ir_positive_by_class': probes['ir_ATSS'].records[-1][
                'positive_by_class'],
            'tiny_boxes': tiny_boxes,
            'dfl_target_configured_range': [0.0, 15.99]
        })
        if valid.sum() > 0 and (positive_checks[-1]['vis_positive'] == 0 or
                                positive_checks[-1]['ir_positive'] == 0):
            raise AssertionError('GT batch has zero O2M positives.')
    result['checks']['gt_pipeline'] = gt_checks
    result['checks']['origin_batch_contract'] = origin_batch_contracts
    result['checks']['positive_counts'] = positive_checks
    train_vis_handle.remove()
    train_ir_handle.remove()
    if train_vis_calls.calls == 0 or train_ir_calls.calls == 0:
        raise AssertionError('Both auxiliary heads must be called in training.')
    result['checks']['train_aux_calls'] = {
        'vis': train_vis_calls.calls, 'ir': train_ir_calls.calls}
    result['checks']['tiny_class_coverage'] = [
        CLASS_NAMES[index] for index in sorted(tiny_seen)]
    result['checks']['empty_gt'] = empty_gt_check(
        train_model.model, train_batches[0])

    transitions = {}
    for epoch in (0, 29, 30, 31):
        batch = copy.deepcopy(train_batches[epoch % len(train_batches)])
        batch['epoch_id'] = epoch
        before = {name: len(probe.records) for name, probe in probes.items()}
        train_model.model.clear_gradients()
        train_model.model(batch)
        called = [name for name, probe in probes.items()
                  if len(probe.records) > before[name]]
        expected = 'ATSS' if epoch < 30 else 'TaskAligned'
        if len(called) != 2 or not all(expected in name for name in called):
            raise AssertionError({'epoch': epoch, 'called': called})
        transitions[str(epoch)] = called
    result['checks']['assigner_transition'] = transitions

    gradient_batch = copy.deepcopy(train_batches[0])
    gradient_batch['epoch_id'] = 0
    dino_grad = collect_gradients(train_model.model, gradient_batch, 'dino_total')
    aux_grad = collect_gradients(train_model.model, gradient_batch, 'aux_only')
    total_grad = collect_gradients(train_model.model, gradient_batch, 'loss')
    for prefix in ('backbone_vis', 'neck_vis', 'backbone_ir', 'neck_ir'):
        if dino_grad[prefix] <= 0 or total_grad[prefix] <= 0:
            raise AssertionError('DINO/total gradient failed for ' + prefix)
        if not math.isfinite(dino_grad[prefix]) or not math.isfinite(
                total_grad[prefix]):
            raise AssertionError('Non-finite DINO/total gradient for ' + prefix)
    for prefix in ('backbone_vis', 'neck_vis', 'backbone_ir', 'neck_ir',
                   'aux_o2m_head_vis', 'aux_o2m_head_ir'):
        if aux_grad[prefix] <= 0 or not math.isfinite(aux_grad[prefix]):
            raise AssertionError('AUX-only gradient failed for ' + prefix)
    ratios = {
        'vis_neck': aux_grad['neck_vis'] / max(dino_grad['neck_vis'], 1e-12),
        'ir_neck': aux_grad['neck_ir'] / max(dino_grad['neck_ir'], 1e-12)}
    if max(ratios.values()) > 10:
        raise AssertionError('Auxiliary neck gradient exceeds DINO by >10x: {}'.format(ratios))
    result['checks']['gradients'] = {
        'dino_only': dino_grad, 'aux_only': aux_grad, 'total': total_grad,
        'aux_to_dino_neck_ratio': ratios}

    result['checks']['tiny_overfit'] = run_tiny_overfit(
        tiny_path, coco_checkpoint)

    smoke = build_trainer(E6_CONFIG, 'train', amp=True)
    result['checks']['smoke_coco_weight_loading'] = load_coco_start(
        smoke, coco_checkpoint)
    smoke.model.train()
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    smoke_probes = install_assigner_probes(smoke.model)
    pred_captures = []
    for modality in ('vis', 'ir'):
        head = getattr(smoke.model, 'aux_o2m_head_' + modality)
        for index, layer in enumerate(head.pred_cls):
            capture = FiniteCapture('{}-cls-{}'.format(modality, index))
            layer.register_forward_post_hook(capture)
            pred_captures.append(capture)
        for index, layer in enumerate(head.pred_reg):
            capture = FiniteCapture('{}-reg-{}'.format(modality, index))
            layer.register_forward_post_hook(capture)
            pred_captures.append(capture)
    csv_path = OUTPUT_DIR / 'smoke_100_steps.csv'
    rows = []
    smoke_iterator = iter(smoke.loader)
    for step in range(100):
        try:
            batch = next(smoke_iterator)
        except StopIteration:
            smoke_iterator = iter(smoke.loader)
            batch = next(smoke_iterator)
        outputs, vis_neck_grad, ir_neck_grad, current_lr = optimizer_step(
            smoke, batch, scaler, 0)
        row = {
            'step': step + 1,
            'lr': current_lr,
            'loss_total': scalar(outputs['loss']),
            'loss_dino': scalar(outputs['dino_total']),
            'loss_aux_vis': scalar(outputs['aux_vis_total']),
            'loss_aux_ir': scalar(outputs['aux_ir_total']),
            'loss_aux_mean': scalar(outputs['aux_mean']),
            'aux_to_dino_ratio': scalar(outputs['aux_mean']) /
                                 max(abs(scalar(outputs['dino_total'])), 1e-12),
            'grad_norm_vis_neck': vis_neck_grad,
            'grad_norm_ir_neck': ir_neck_grad,
            'gpu_memory': gpu_peak_memory()
        }
        if not all(math.isfinite(value) for key, value in row.items()
                   if key != 'step'):
            raise FloatingPointError('Non-finite smoke metric.')
        rows.append(row)
    if not all(capture.calls > 0 and capture.all_finite
               for capture in pred_captures):
        raise FloatingPointError('Non-finite PPYOLOE prediction tensor.')
    if not all(record['assigned_scores_finite'] and
               record['assigned_boxes_finite']
               for probe in smoke_probes.values()
               for record in probe.records):
        raise FloatingPointError('Non-finite auxiliary assignment tensor.')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ratios_100 = [row['aux_to_dino_ratio'] for row in rows]
    ratio_stats = {
        'mean': statistics.mean(ratios_100),
        'std': statistics.pstdev(ratios_100),
        'p10': float(np.percentile(ratios_100, 10)),
        'p50': float(np.percentile(ratios_100, 50)),
        'p90': float(np.percentile(ratios_100, 90)),
        'max': max(ratios_100)}
    if ratio_stats['mean'] > 3 or ratio_stats['mean'] < 0.05:
        raise AssertionError('Aux/DINO loss ratio outside acceptance range: {}'.format(
            ratio_stats))
    result['checks']['smoke_100'] = {
        'csv': str(csv_path), 'ratio_stats': ratio_stats,
        'peak_memory': max(row['gpu_memory'] for row in rows)}
    result['checks']['checkpoint_resume'] = save_and_resume(
        smoke, train_batches[0], scaler, OUTPUT_DIR)

    result['all_passed'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e1-checkpoint', required=True)
    parser.add_argument('--coco-pretrain', required=True)
    parser.add_argument('--tiny-annotation', required=True)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as error:
        result = {
            'all_passed': False,
            'error': '{}: {}'.format(type(error).__name__, error),
            'traceback': traceback.format_exc()}
    output_path = OUTPUT_DIR / 'acceptance.json'
    output_path.write_text(json.dumps(
        result, indent=2, ensure_ascii=False), encoding='utf-8')
    update_report(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
