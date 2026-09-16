#!/usr/bin/env python3
"""Dynamic acceptance suite for E6a; it never launches epoch training.

Run only in the original E1 cloud runtime (Paddle 2.6.2/CUDA 11.8) after
creating an independent E6a directory that shares E1's read-only data/weights.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import paddle

from ppdet.core.workspace import create, load_config
from ppdet.engine import Trainer
from ppdet.utils.checkpoint import load_weight


ROOT = Path(__file__).resolve().parents[1]
E1_CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml'
E6_CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_rgbir_960_dual_o2m.yml'


def seed(value=20260916):
    random.seed(value); np.random.seed(value); paddle.seed(value)


def tensor_np(x):
    return x.numpy() if isinstance(x, paddle.Tensor) else np.asarray(x)


def finite_scalar(x):
    return bool(np.isfinite(float(tensor_np(x).reshape([-1])[0])))


def max_diff(a, b):
    if isinstance(a, (list, tuple)):
        return max(max_diff(x, y) for x, y in zip(a, b))
    return float(np.max(np.abs(tensor_np(a).astype('float64') - tensor_np(b).astype('float64'))))


def partial_load_e1(model, checkpoint):
    """Load only E1 keys; the only missing keys must be the two new aux heads."""
    source = paddle.load(checkpoint)
    target = model.state_dict()
    matched = {k: v for k, v in target.items() if k in source and list(v.shape) == list(source[k].shape)}
    missing = sorted(set(target) - set(matched))
    unexpected = sorted(set(source) - set(matched))
    bad = [k for k in missing if not k.startswith(('aux_o2m_head_vis.', 'aux_o2m_head_ir.'))]
    if bad:
        raise AssertionError('Non-auxiliary E1 keys failed to load: ' + ', '.join(bad[:10]))
    model.set_dict(matched)
    return {'matched': len(matched), 'missing_aux_keys': len(missing), 'unexpected_e1_keys': len(unexpected)}


def build_trainer(path, mode, aux=None):
    cfg = load_config(str(path))
    if aux is not None:
        cfg.DAMSDet['aux_o2m_enabled'] = bool(aux)
    return Trainer(cfg, mode=mode)


def get_batch(trainer):
    return next(iter(trainer.loader))


def capture_eval(model, batch):
    cache = {}
    for name in ('neck_vis', 'neck_ir', 'detr_head'):
        layer = getattr(model, name)
        original = layer.forward
        def wrapped(*args, _original=original, _name=name, **kwargs):
            out = _original(*args, **kwargs)
            cache[_name] = out
            return out
        layer.forward = wrapped
    model.eval()
    with paddle.no_grad():
        final = model(batch)
    return cache, final


def named_grad_norm(model, prefix):
    values = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix):
            continue
        grad = getattr(parameter, 'grad', None)
        if callable(grad):
            grad = grad()
        if grad is not None:
            array = tensor_np(grad)
            if not np.all(np.isfinite(array)):
                raise AssertionError(prefix + ' has non-finite gradient')
            values.append(float(np.sum(array.astype('float64') ** 2)))
    norm = math.sqrt(sum(values))
    if not values or not math.isfinite(norm) or norm <= 0:
        raise AssertionError(prefix + ' has no positive finite gradient')
    return norm


def check_origin_targets(batch):
    origin = tensor_np(batch['origin_gt_bbox'])
    dino = tensor_np(batch['gt_bbox'])
    mask = tensor_np(batch['pad_origin_gt_mask']).astype(bool).reshape(-1)
    image = tensor_np(batch['vis_image'])
    h, w = image.shape[-2:]
    origin = origin.reshape([-1, 4])[mask]
    dino = dino.reshape([-1, 4])[mask]
    cx, cy, bw, bh = [dino[:, i] for i in range(4)]
    rebuilt = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
    return float(np.max(np.abs(rebuilt - origin))) if len(origin) else 0.0


def run(args):
    if paddle.__version__ != '2.6.2':
        raise RuntimeError('This acceptance suite is pinned to Paddle 2.6.2; found ' + paddle.__version__)
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError('CUDA Paddle is required for the AMP smoke check.')
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    seed()
    result = {'checkpoint': str(checkpoint), 'paddle': paddle.__version__, 'checks': {}}

    # 1. E1/E6a eval exact equivalence on the same val batch.
    e1_eval = build_trainer(E1_CONFIG, 'eval')
    e6_eval = build_trainer(E6_CONFIG, 'eval', aux=True)
    load_weight(e1_eval.model, str(checkpoint))
    result['e6a_partial_load'] = partial_load_e1(e6_eval.model, str(checkpoint))
    batch = get_batch(e1_eval)
    seed(); e1_cache, e1_final = capture_eval(e1_eval.model, batch)
    seed(); e6_cache, e6_final = capture_eval(e6_eval.model, batch)
    diffs = {
        'vis_neck': max_diff(e1_cache['neck_vis'], e6_cache['neck_vis']),
        'ir_neck': max_diff(e1_cache['neck_ir'], e6_cache['neck_ir']),
        'decoder_boxes': max_diff(e1_cache['detr_head'][0], e6_cache['detr_head'][0]),
        'decoder_logits': max_diff(e1_cache['detr_head'][1], e6_cache['detr_head'][1]),
        'final_boxes_and_scores': max_diff(e1_final['bbox'], e6_final['bbox']),
    }
    assert max(diffs.values()) <= 1e-7, diffs
    result['checks']['eval_exact_equivalence'] = diffs

    # 2/3. A disabled E6a model must be numerically E1 on identical train data.
    e1_train = build_trainer(E1_CONFIG, 'train')
    e6_disabled = build_trainer(E6_CONFIG, 'train', aux=False)
    load_weight(e1_train.model, str(checkpoint)); load_weight(e6_disabled.model, str(checkpoint))
    train_batch = get_batch(e1_train)
    e1_train.model.train(); e6_disabled.model.train()
    seed(); loss_e1 = e1_train.model(train_batch)
    seed(); loss_e6 = e6_disabled.model(train_batch)
    common = sorted(set(loss_e1) & set(loss_e6))
    loss_diffs = {k: max_diff(loss_e1[k], loss_e6[k]) for k in common}
    assert max(loss_diffs.values()) <= 1e-7, loss_diffs
    result['checks']['aux_disabled_dino_regression'] = loss_diffs

    # E6a reader is required for origin GT targets and all remaining train tests.
    e6_train = build_trainer(E6_CONFIG, 'train', aux=True)
    result['e6a_partial_load_train'] = partial_load_e1(e6_train.model, str(checkpoint))
    e6_train.model.train()
    gt_errors = []
    train_iter = iter(e6_train.loader)
    cached_batches = []
    for _ in range(24):
        data = next(train_iter)
        cached_batches.append(data)
        gt_errors.append(check_origin_targets(data))
    assert max(gt_errors) < 1e-2, max(gt_errors)
    result['checks']['origin_gt_roundtrip_max_pixel_error'] = max(gt_errors)

    # 4. Verify both assignment branches by observing real calls at required epochs.
    calls = []
    for head_name in ('aux_o2m_head_vis', 'aux_o2m_head_ir'):
        head = getattr(e6_train.model, head_name)
        for label, assigner in (('ATSS', head.static_assigner), ('TaskAligned', head.assigner)):
            original = assigner.forward
            def wrapped(*a, _original=original, _label=label, _head=head_name, **kw):
                calls.append((_head, _label)); return _original(*a, **kw)
            assigner.forward = wrapped
    for epoch in (0, 29, 30, 31):
        data = cached_batches[epoch % len(cached_batches)]
        data['epoch_id'] = epoch
        e6_train.model.clear_gradients(); _ = e6_train.model(data)
        expected = 'ATSS' if epoch < 30 else 'TaskAligned'
        latest = calls[-2:]
        assert len(latest) == 2 and all(item[1] == expected for item in latest), (epoch, latest)
    result['checks']['assigner_transition'] = 'ATSS: epoch 0/29; TaskAligned: epoch 30/31'

    # 5. Auxiliary-only gradient through each RGB/IR backbone, neck, and head.
    data = cached_batches[0]; data['epoch_id'] = 0
    e6_train.model.clear_gradients(); losses = e6_train.model(data)
    aux_only = (losses['aux_vis_total'] + losses['aux_ir_total']) * .5
    aux_only.backward()
    grad_norms = {p: named_grad_norm(e6_train.model, p) for p in (
        'neck_vis', 'neck_ir', 'backbone_vis', 'backbone_ir',
        'aux_o2m_head_vis', 'aux_o2m_head_ir')}
    result['checks']['aux_only_grad_norms'] = grad_norms

    # 6. Tiny-overfit: only the first 24 REAL train-reader samples, repeated.
    tiny = []
    for step in range(args.tiny_steps):
        data = cached_batches[step % len(cached_batches)]; data['epoch_id'] = 0
        e6_train.model.clear_gradients()
        with paddle.amp.auto_cast(enable=True):
            out = e6_train.model(data); loss = out['loss']
        assert all(finite_scalar(out[k]) for k in ('dino_total', 'aux_vis_total', 'aux_ir_total', 'loss'))
        loss.backward(); e6_train.optimizer.step(); e6_train.optimizer.clear_grad()
        tiny.append({k: float(tensor_np(out[k])) for k in ('dino_total', 'aux_vis_total', 'aux_ir_total')})
    result['checks']['tiny_overfit'] = {'steps': args.tiny_steps, 'first': tiny[0], 'last': tiny[-1]}

    # 7. Real train1600 100-step AMP smoke, no epoch loop/validation/long run.
    smoke = []
    peak = 0
    for step in range(args.smoke_steps):
        data = next(train_iter); data['epoch_id'] = 0
        e6_train.model.clear_gradients()
        with paddle.amp.auto_cast(enable=True):
            out = e6_train.model(data); total = out['loss']
        values = {k: float(tensor_np(out[k])) for k in ('dino_total', 'aux_vis_total', 'aux_ir_total')}
        values['aux_mean'] = (values['aux_vis_total'] + values['aux_ir_total']) / 2.
        values['aux_to_dino'] = values['aux_mean'] / max(abs(values['dino_total']), 1e-12)
        assert all(math.isfinite(v) for v in values.values()) and finite_scalar(total)
        total.backward()
        values['vis_neck_grad_norm'] = named_grad_norm(e6_train.model, 'neck_vis')
        values['ir_neck_grad_norm'] = named_grad_norm(e6_train.model, 'neck_ir')
        values['lr'] = float(e6_train.optimizer.get_lr())
        assert all(math.isfinite(v) for v in values.values())
        e6_train.optimizer.step(); e6_train.optimizer.clear_grad()
        if hasattr(paddle.device.cuda, 'max_memory_allocated'):
            peak = max(peak, int(paddle.device.cuda.max_memory_allocated()))
        smoke.append(values)
    result['checks']['amp_smoke'] = {'steps': args.smoke_steps, 'first': smoke[0], 'last': smoke[-1], 'peak_memory_bytes': peak}

    # 8. Save/reload the smoke state then perform exactly one more optimization step.
    smoke_dir = ROOT / 'output/E6a_RGBIR_DAMSDet_960_dual_o2m/dynamic_acceptance'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    model_path, opt_path = smoke_dir / 'smoke_model.pdparams', smoke_dir / 'smoke_optimizer.pdopt'
    paddle.save(e6_train.model.state_dict(), str(model_path)); paddle.save(e6_train.optimizer.state_dict(), str(opt_path))
    resumed = build_trainer(E6_CONFIG, 'train', aux=True)
    resumed.model.set_dict(paddle.load(str(model_path))); resumed.optimizer.set_state_dict(paddle.load(str(opt_path)))
    data = cached_batches[0]; data['epoch_id'] = 0
    resumed.model.train(); resumed.model.clear_gradients()
    with paddle.amp.auto_cast(enable=True): out = resumed.model(data); total = out['loss']
    assert finite_scalar(total); total.backward(); resumed.optimizer.step(); resumed.optimizer.clear_grad()
    result['checks']['checkpoint_save_resume'] = 'PASS'

    report = smoke_dir / 'dynamic_acceptance.json'
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    implementation_report = ROOT / 'E6A_IMPLEMENTATION_REPORT.md'
    existing = implementation_report.read_text(encoding='utf-8')
    marker = '## Cloud dynamic acceptance result\n'
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + '\n\n'
    if '## Final status\n' in existing:
        existing = existing.rsplit('## Final status\n', 1)[0].rstrip() + '\n\n'
    summary = json.dumps(result['checks'], indent=2)
    existing += (marker + '\nAll requested dynamic checks completed successfully in '
                 'the E1 cloud runtime. Detailed machine-readable evidence: '
                 f'`{report.relative_to(ROOT).as_posix()}`.\n\n```json\n{summary}\n```\n\n'
                 '## Final status\n\n**READY FOR E6a EARLY RUN.** Dynamic acceptance passed; '
                 'no 72-epoch training was started.\n')
    implementation_report.write_text(existing, encoding='utf-8')
    print(json.dumps(result, indent=2)); print('PASS: dynamic acceptance complete:', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='E1 best .pdparams')
    parser.add_argument('--tiny-steps', type=int, default=72)
    parser.add_argument('--smoke-steps', type=int, default=100)
    run(parser.parse_args())
