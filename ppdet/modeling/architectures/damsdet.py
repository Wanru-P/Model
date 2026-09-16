# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DAMSDet']

@register
class DAMSDet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_vis=None,
                 neck_ir=None,
                 depth_encoder=None,
                 depth_gating=None,
                 aux_o2m_head_vis=None,
                 aux_o2m_head_ir=None,
                 aux_o2m_enabled=False,
                 aux_o2m_weight=1.0,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DAMSDet, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_vis = neck_vis
        self.neck_ir = neck_ir
        self.depth_encoder = depth_encoder
        self.depth_gating = depth_gating
        self.aux_o2m_head_vis = aux_o2m_head_vis
        self.aux_o2m_head_ir = aux_o2m_head_ir
        self.aux_o2m_enabled = bool(aux_o2m_enabled)
        self.aux_o2m_weight = float(aux_o2m_weight)
        if self.aux_o2m_enabled and (self.aux_o2m_head_vis is None or
                                     self.aux_o2m_head_ir is None):
            raise ValueError(
                'E6a requires independent VIS and IR auxiliary heads.')
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        depth_encoder = create(cfg['depth_encoder']) \
            if cfg.get('depth_encoder', None) else None
        depth_gating = None
        if depth_encoder is not None and not cfg.get('depth_gating', None):
            raise ValueError('depth_encoder requires depth_gating.')
        if cfg.get('depth_gating', None):
            if depth_encoder is None:
                raise ValueError('depth_gating requires depth_encoder.')
            depth_gating = create(
                cfg['depth_gating'],
                rgb_input_shape=backbone_vis.out_shape,
                depth_input_shape=depth_encoder.out_shape)
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_vis = create(cfg['neck_vis'], **kwargs) if cfg['neck_vis'] else None
        neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None
        # transformer
        if neck_vis is not None:
            kwargs = {'input_shape': neck_vis.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        # Match official RT-DETRv3 construction order: auxiliary heads are
        # instantiated only after the complete DETR main path. This preserves
        # E1 RNG initialization for main-path tensors that COCO does not load.
        aux_o2m_enabled = bool(cfg.get('aux_o2m_enabled', False))
        aux_o2m_head_vis = None
        aux_o2m_head_ir = None
        if aux_o2m_enabled:
            if neck_vis is None or neck_ir is None:
                raise ValueError(
                    'E6a auxiliary heads require both HybridEncoders.')
            aux_o2m_head_vis = create(
                cfg['aux_o2m_head_vis'], input_shape=neck_vis.out_shape)
            aux_o2m_head_ir = create(
                cfg['aux_o2m_head_ir'], input_shape=neck_ir.out_shape)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_vis": neck_vis,
            "neck_ir": neck_ir,
            "depth_encoder": depth_encoder,
            "depth_gating": depth_gating,
            "aux_o2m_head_vis": aux_o2m_head_vis,
            "aux_o2m_head_ir": aux_o2m_head_ir,
            "aux_o2m_enabled": aux_o2m_enabled,
            "aux_o2m_weight": cfg.get('aux_o2m_weight', 1.0)
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)

        # Depth is an auxiliary residual path.  The original RGB/IR backbones,
        # HybridEncoders, MCQS, MDCA and decoder remain unchanged.
        if self.depth_encoder is not None:
            if 'depth_image' not in self.inputs or \
                    'depth_valid_mask' not in self.inputs:
                raise KeyError(
                    'Quality-aware depth DAMSDet requires depth_image and '
                    'depth_valid_mask in every sample.')
            depth_feats = self.depth_encoder(self.inputs)
            vis_body_feats, ir_body_feats = self.depth_gating(
                vis_body_feats,
                ir_body_feats,
                depth_feats,
                self.inputs['depth_valid_mask'])

        # Neck
        if self.neck_vis is not None:
            vis_body_feats = self.neck_vis(vis_body_feats)
            ir_body_feats = self.neck_ir(ir_body_feats)

        if self.training and self.aux_o2m_enabled:
            if len(vis_body_feats) != 3 or len(ir_body_feats) != 3:
                raise ValueError('E6a auxiliary heads require P3/P4/P5.')
            vis_channels = [feature.shape[1] for feature in vis_body_feats]
            ir_channels = [feature.shape[1] for feature in ir_body_feats]
            if vis_channels != [256, 256, 256] or \
                    ir_channels != [256, 256, 256]:
                raise ValueError(
                    'E6a HybridEncoder channels must be [256, 256, 256].')

        pad_mask = self.inputs.get('pad_mask', None)

        out_transformer = self.transformer(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            dino_total = paddle.add_n(
                [v for k, v in detr_losses.items() if 'log' not in k])
            detr_losses['loss'] = dino_total
            if self.aux_o2m_enabled:
                aux_vis = self.aux_o2m_head_vis(vis_body_feats, self.inputs)
                aux_ir = self.aux_o2m_head_ir(ir_body_feats, self.inputs)
                aux_mean = (aux_vis['loss'] + aux_ir['loss']) * 0.5
                detr_losses['loss'] = (
                    dino_total + self.aux_o2m_weight * aux_mean)
                detr_losses.update({
                    'dino_total': dino_total,
                    'aux_vis_total': aux_vis['loss'],
                    'aux_ir_total': aux_ir['loss'],
                    'aux_mean': aux_mean,
                    'aux_vis_cls': aux_vis['loss_cls'],
                    'aux_vis_iou': aux_vis['loss_iou'],
                    'aux_vis_dfl': aux_vis['loss_dfl'],
                    'aux_ir_cls': aux_ir['loss_cls'],
                    'aux_ir_iou': aux_ir['loss_iou'],
                    'aux_ir_dfl': aux_ir['loss_dfl']
                })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, None)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
