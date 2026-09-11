# Copyright (c) 2026
#
# Lightweight depth support for DAMSDet.  Depth is deliberately kept as an
# auxiliary geometric cue: it never replaces either of the two original
# ResNet50-vd streams.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, KaimingNormal

from ppdet.core.workspace import register
from ..shape_spec import ShapeSpec

__all__ = ['LiteDepthEncoder', 'QualityAwareDepthGating']


class ConvBNAct(nn.Layer):
    """Small convolution block used only by the auxiliary depth branch."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Layer):

    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableBlock, self).__init__()
        self.depthwise = ConvBNAct(in_channels,
                                   in_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   groups=in_channels)
        self.pointwise = ConvBNAct(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@register
class LiteDepthEncoder(nn.Layer):
    """Encode one-channel depth into stride-8/16/32 (P3/P4/P5) features.

    The encoder is intentionally much smaller than either RGB/IR backbone.
    Invalid pixels are zeroed before feature extraction; their spatial mask is
    also passed independently to :class:`QualityAwareDepthGating`.
    """

    def __init__(self, channels=(32, 48, 64, 128, 256)):
        super(LiteDepthEncoder, self).__init__()
        if len(channels) != 5:
            raise ValueError(
                'LiteDepthEncoder channels must contain 5 values.')
        c1, c2, c3, c4, c5 = channels
        self.stem = ConvBNAct(1, c1, kernel_size=3, stride=2)
        self.stage2 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.stage3 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.stage4 = DepthwiseSeparableBlock(c3, c4, stride=2)
        self.stage5 = DepthwiseSeparableBlock(c4, c5, stride=2)
        self._out_channels = [c3, c4, c5]

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=c, stride=s)
            for c, s in zip(self._out_channels, [8, 16, 32])
        ]

    def forward(self, inputs):
        depth = inputs['depth_image']
        valid = inputs.get('depth_valid_mask', None)
        if valid is not None:
            depth = depth * valid.astype(depth.dtype)
        x = self.stem(depth)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return [p3, p4, p5]


class QualityHead(nn.Layer):
    """Predict a spatial depth-quality probability from feature + valid mask."""

    def __init__(self, channels, hidden_channels, gate_bias=-2.0):
        super(QualityHead, self).__init__()
        self.conv = ConvBNAct(channels + 1,
                              hidden_channels,
                              kernel_size=3,
                              stride=1)
        self.logits = nn.Conv2D(
            hidden_channels,
            1,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=ParamAttr(initializer=Constant(gate_bias)))

    def forward(self, depth_feature, valid_mask):
        x = paddle.concat([depth_feature, valid_mask], axis=1)
        return F.sigmoid(self.logits(self.conv(x))) * valid_mask


@register
class QualityAwareDepthGating(nn.Layer):
    """Inject reliable depth into the two untouched DAMSDet backbone streams.

    For scale ``i``::

        R_i' = R_i + alpha_i * Q_i * P_i(D_i)
        I_i' = I_i + beta_i  * Q_i * P_i(D_i)

    ``P_i`` is zero-initialized by default, so the complete model starts as the
    original RGB+IR DAMSDet.  Alpha/beta remain non-zero to let the projection
    receive gradients from the first optimization step.
    """

    def __init__(self,
                 rgb_channels=(512, 1024, 2048),
                 depth_channels=(64, 128, 256),
                 gate_hidden_channels=(32, 32, 64),
                 alpha_init=1.0,
                 beta_init=1.0,
                 gate_bias=-2.0,
                 zero_init_projection=True):
        super(QualityAwareDepthGating, self).__init__()
        if not (len(rgb_channels) == len(depth_channels) ==
                len(gate_hidden_channels) == 3):
            raise ValueError(
                'QualityAwareDepthGating expects exactly P3/P4/P5.')

        projection_initializer = Constant(0.0) if zero_init_projection \
            else KaimingNormal()
        self.depth_projections = nn.LayerList()
        self.quality_heads = nn.LayerList()
        self.alpha = nn.ParameterList()
        self.beta = nn.ParameterList()

        for rgb_c, depth_c, hidden_c in zip(rgb_channels, depth_channels,
                                            gate_hidden_channels):
            self.depth_projections.append(
                nn.Conv2D(
                    depth_c,
                    rgb_c,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=projection_initializer),
                    bias_attr=ParamAttr(initializer=Constant(0.0))))
            self.quality_heads.append(
                QualityHead(depth_c, hidden_c, gate_bias=gate_bias))
            self.alpha.append(
                self.create_parameter(
                    shape=[1], default_initializer=Constant(alpha_init)))
            self.beta.append(
                self.create_parameter(shape=[1],
                                      default_initializer=Constant(beta_init)))

    @classmethod
    def from_config(cls, cfg, rgb_input_shape, depth_input_shape):
        return {
            'rgb_channels': [shape.channels for shape in rgb_input_shape],
            'depth_channels': [shape.channels for shape in depth_input_shape]
        }

    @staticmethod
    def _resize_like(x, reference, mode):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode=mode)
        return x

    def forward(self, vis_features, ir_features, depth_features, valid_mask):
        if not (len(vis_features) == len(ir_features) == len(depth_features) ==
                3):
            raise ValueError(
                'DAMSDet depth gating requires three feature levels.')

        gated_vis, gated_ir = [], []
        valid_mask = valid_mask.astype(depth_features[0].dtype)
        for i, (vis, ir, depth) in enumerate(
                zip(vis_features, ir_features, depth_features)):
            depth = self._resize_like(depth, vis, mode='bilinear')
            # Average pooling expresses how much valid depth supports each
            # feature cell (0 = invalid, 1 = fully valid), rather than sampling
            # one arbitrary source pixel with nearest-neighbour interpolation.
            mask = F.adaptive_avg_pool2d(valid_mask,
                                         output_size=vis.shape[-2:])
            quality = self.quality_heads[i](depth, mask)
            residual = quality * self.depth_projections[i](depth)
            gated_vis.append(vis + self.alpha[i] * residual)
            gated_ir.append(ir + self.beta[i] * residual)
        return gated_vis, gated_ir
