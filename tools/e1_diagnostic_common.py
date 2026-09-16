"""Read-only helpers shared by E1 diagnostic scripts.

The helpers intentionally reconstruct existing evaluation operations without
changing any object in ``ppdet``.  They are not part of the training path.
"""

from __future__ import print_function

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def require_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError('{} does not exist: {}'.format(label, path))
    return path


def build_eval_trainer(config_path, checkpoint_path, device):
    import paddle
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer

    config_path = require_file(config_path, 'config')
    checkpoint_path = require_file(checkpoint_path, 'checkpoint')
    paddle.set_device(device)
    cfg = load_config(str(config_path))
    cfg.use_gpu = device.startswith('gpu')
    cfg.use_npu = False
    cfg.use_xpu = False
    cfg.use_mlu = False
    cfg.fleet = False
    cfg.amp = True
    trainer = Trainer(cfg, mode='eval')
    trainer.load_weights(str(checkpoint_path))
    trainer.model.eval()
    return cfg, trainer


def runtime_info(device):
    import paddle
    info = {
        'paddle': paddle.__version__,
        'cuda': str(paddle.version.cuda()),
        'device': device,
        'gpu': 'unavailable',
    }
    if device.startswith('gpu'):
        try:
            info['gpu'] = paddle.device.cuda.get_device_name(0)
        except Exception:
            info['gpu'] = 'GPU 0'
    return info


def load_annotation_maps(annotation_path):
    payload = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    image_by_id = {int(item['id']): item for item in payload['images']}
    annotations_by_image = {}
    for annotation in payload.get('annotations', []):
        if annotation.get('iscrowd', 0):
            continue
        annotations_by_image.setdefault(int(annotation['image_id']), []).append(
            annotation)
    categories = sorted(payload['categories'], key=lambda item: int(item['id']))
    cat_id_by_class = {index: int(item['id'])
                       for index, item in enumerate(categories)}
    class_name_by_class = {index: item['name']
                           for index, item in enumerate(categories)}
    class_by_cat_id = {category_id: class_id
                       for class_id, category_id in cat_id_by_class.items()}
    return (payload, image_by_id, annotations_by_image, cat_id_by_class,
            class_name_by_class, class_by_cat_id)


def finite_or_fail(name, tensor):
    import paddle
    if not bool(paddle.isfinite(tensor).all().numpy().item()):
        raise RuntimeError('Non-finite tensor in diagnostic: {}'.format(name))


def run_encoder_topk(model, batch):
    """Reproduce production encoder TopK with the same tensors and ordering."""
    import paddle

    vis_body = model.backbone_vis(batch, 1)
    ir_body = model.backbone_ir(batch, 2)
    vis_feats = model.neck_vis(vis_body) if model.neck_vis is not None else vis_body
    ir_feats = model.neck_ir(ir_body) if model.neck_ir is not None else ir_body
    transformer = model.transformer
    fused_feats = vis_feats + ir_feats
    memory, spatial_shapes, level_start = transformer._get_encoder_visir_input(
        fused_feats)
    if transformer.training or transformer.eval_size is None:
        anchors, anchor_valid = transformer._generate_anchors(spatial_shapes)
    else:
        anchors, anchor_valid = transformer.anchors, transformer.valid_mask
    masked_memory = paddle.where(anchor_valid, memory, paddle.to_tensor(0.0))
    output_memory = transformer.enc_output(masked_memory)
    enc_logits = transformer.enc_score_head(output_memory)
    enc_boxes = transformer.enc_bbox_head(output_memory) + anchors
    topk_logits, topk_indices = paddle.topk(
        enc_logits.max(-1), transformer.num_queries, axis=1)
    batch_indices = paddle.arange(end=topk_indices.shape[0], dtype=topk_indices.dtype)
    batch_indices = batch_indices.unsqueeze(-1).tile([1, transformer.num_queries])
    gather_indices = paddle.stack([batch_indices, topk_indices], axis=-1)
    selected_logits = paddle.gather_nd(enc_logits, gather_indices)
    selected_boxes = paddle.gather_nd(enc_boxes, gather_indices)
    finite_or_fail('encoder logits', enc_logits)
    finite_or_fail('encoder boxes at selected TopK', selected_boxes)
    finite_or_fail('TopK logits', topk_logits)
    if not bool(paddle.allclose(topk_logits, selected_logits.max(-1)).numpy().item()):
        raise RuntimeError('Reconstructed TopK scores differ from gathered logits')
    return {
        'vis_feats': vis_feats,
        'ir_feats': ir_feats,
        'memory': memory,
        'spatial_shapes': spatial_shapes,
        'level_start': level_start,
        'anchor_valid': anchor_valid,
        'enc_logits': enc_logits,
        'enc_boxes': enc_boxes,
        'topk_logits': topk_logits,
        'topk_indices': topk_indices,
        'selected_logits': selected_logits,
    }


def make_feature_pad_masks(input_pad_mask, spatial_shapes):
    """Nearest-downsample [B,H,W] real=1/pad=0 mask in memory order."""
    import paddle.nn.functional as F

    if input_pad_mask.ndim == 3:
        input_pad_mask = input_pad_mask.unsqueeze(1)
    if input_pad_mask.ndim != 4:
        raise RuntimeError('Unexpected pad_mask shape: {}'.format(
            tuple(input_pad_mask.shape)))
    level_masks = []
    for shape in spatial_shapes:
        height, width = int(shape[0]), int(shape[1])
        resized = F.interpolate(input_pad_mask, size=[height, width], mode='nearest')
        level_masks.append((resized > 0.5).reshape([resized.shape[0], -1]))
    memory_valid = __import__('paddle').concat(level_masks, axis=1)
    return level_masks, memory_valid


def run_decoder_and_postprocess(model, batch):
    """Run the exact eval model path once and expose raw decoder and stage-2."""
    import paddle

    vis_body = model.backbone_vis(batch, 1)
    ir_body = model.backbone_ir(batch, 2)
    vis_feats = model.neck_vis(vis_body) if model.neck_vis is not None else vis_body
    ir_feats = model.neck_ir(ir_body) if model.neck_ir is not None else ir_body
    transformer_out = model.transformer(
        None, vis_feats, ir_feats, batch.get('pad_mask'), batch)
    raw_boxes, raw_logits, _ = model.detr_head(transformer_out, None)
    stage2_boxes, stage2_num, _ = model.post_process(
        (raw_boxes, raw_logits, None), batch['im_shape'], batch['scale_factor'],
        paddle.shape(batch['vis_image'])[2:])
    finite_or_fail('raw decoder boxes', raw_boxes)
    finite_or_fail('raw decoder logits', raw_logits)
    finite_or_fail('stage2 boxes', stage2_boxes)
    return raw_boxes, raw_logits, stage2_boxes, stage2_num


def tensor_to_numpy(tensor):
    return np.asarray(tensor.numpy())


def decode_raw_query_boxes(raw_boxes, im_shape, scale_factor, pad_shape):
    """Mirror DETRPostProcess lines 505--518 without its Top300 operation."""
    import paddle
    from ppdet.modeling.transformers.utils import bbox_cxcywh_to_xyxy

    boxes = bbox_cxcywh_to_xyxy(raw_boxes)
    origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
    out_shape = pad_shape.astype('float32') / im_shape * origin_shape
    out_shape = out_shape.flip(1).tile([1, 2]).unsqueeze(1)
    return boxes * out_shape


def records_from_stage2(stage2_boxes, stage2_num, image_ids, cat_id_by_class):
    records = []
    arrays = tensor_to_numpy(stage2_boxes).reshape([-1, 6])
    counts = tensor_to_numpy(stage2_num).astype(np.int64).reshape(-1)
    ids = tensor_to_numpy(image_ids).astype(np.int64).reshape(-1)
    offset = 0
    for image_id, count in zip(ids, counts):
        rows = arrays[offset:offset + int(count)]
        offset += int(count)
        for rank, row in enumerate(rows, start=1):
            class_id = int(row[0])
            if class_id < 0:
                continue
            x1, y1, x2, y2 = [float(value) for value in row[2:6]]
            records.append({
                'image_id': int(image_id),
                'category_id': int(cat_id_by_class[class_id]),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(row[1]),
                'query_rank': int(rank),
                'class_id': class_id,
            })
    return records


def stage2_rows_by_image(records):
    result = {}
    for record in records:
        result.setdefault(int(record['image_id']), []).append(record)
    for rows in result.values():
        rows.sort(key=lambda item: (-item['score'], item['query_rank']))
    return result


def percentiles(values, percentiles_list):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {str(item): float('nan') for item in percentiles_list}
    return {str(item): float(np.percentile(values, item))
            for item in percentiles_list}


def summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {'count': 0, 'mean': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'p10': float('nan'), 'p50': float('nan'),
                'p90': float('nan'), 'p95': float('nan'), 'max': float('nan')}
    return {
        'count': int(values.size), 'mean': float(values.mean()),
        'std': float(values.std()), 'min': float(values.min()),
        'p10': float(np.percentile(values, 10)),
        'p50': float(np.percentile(values, 50)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)), 'max': float(values.max())}
