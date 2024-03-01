# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_norm_layer, constant_init,
                      normal_init)
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from .pac_siml import PacConv2d
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import DyReLU

# Reference:
# https://github.com/microsoft/DynamicHead
# https://github.com/jshilong/SEPC


class PACCONV(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = PacConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), x.contiguous(), mask=mask)
        if self.with_norm:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.mask_dim = 256

        self.spatial_conv_high = PACCONV(in_channels, out_channels)
        #self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = PACCONV(in_channels, out_channels)
        self.spatial_conv_low = PACCONV(in_channels, out_channels, stride=2)
        self.spatial_conv_mask = nn.Conv2d(
            in_channels, self.mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_mask, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            mask = self.spatial_conv_mask(x[level]).sigmoid()
            mid_feat = self.spatial_conv_mid(x[level], mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                mask_l = F.interpolate(mask,size=x[level - 1].shape[-2:],mode='bilinear', align_corners=True)
                low_feat = self.spatial_conv_low(x[level - 1], mask_l)
                sum_feat = sum_feat + \
                    low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                mask_h = F.interpolate(mask, size=x[level + 1].shape[-2:], mode='bilinear', align_corners=True)
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], mask_h),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat = sum_feat + high_feat * \
                    self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs


@NECKS.register_module()
class DyHeadPAC(BaseModule):
    """DyHead neck consisting of multiple DyHead Blocks.

    See `Dynamic Head: Unifying Object Detection Heads with Attentions
    <https://arxiv.org/abs/2106.08322>`_ for details.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int, optional): Number of DyHead Blocks. Default: 6.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=6,
                 zero_init_offset=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        dyhead_blocks = []
        for i in range(num_blocks):#4
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(
                DyHeadBlock(
                    in_channels,
                    self.out_channels))
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, (tuple, list))
        outs = self.dyhead_blocks(inputs)
        return tuple(outs)
