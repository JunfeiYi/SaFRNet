# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, build_norm_layer
from mmcv.runner import force_fp32
from torch.nn import functional as F
from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmdet.models.utils.batch_drop import Patch_Drop
#from .pac_siml import PacConv2d
from mmcv.ops import BorderAlign
from mmcv.ops import deform_conv2d
from ..utils.se_layer import DyReLU
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d


class PatchStripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self):
        super(PatchStripPooling, self).__init__()
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.patch_ratio = 0.2
        # bilinear interpolate options
    def proj(self,x, patch_size):
        proj = F.unfold(x, kernel_size=patch_size, stride=patch_size).cuda()
        return proj

    def patch_size(self, h, w):
        return min(int(h * self.patch_ratio) + 1, int(w * self.patch_ratio) + 1)

    def re_proj(self,x, patch_size, output_size):
        re_proj = F.fold(x, output_size=output_size,kernel_size=patch_size, stride=patch_size).cuda()
        return re_proj

    def hw_conv(self, x, feat_channels):
        hw_conv = ConvModule(
            feat_channels,
            feat_channels,
            3,
            stride=1,
            padding=1).cuda()
        dyrelu = DyReLU(feat_channels).cuda()
        return dyrelu(hw_conv(x))

    def h_conv(self, x, feat_channels):
        h_conv = nn.Conv2d(feat_channels, feat_channels, (1, 3), 1, (0, 1), bias=True).cuda()
        return h_conv(x)

    def w_conv(self, x, feat_channels):
        w_conv = nn.Conv2d(feat_channels, feat_channels, (1, 3), 1, (0, 1), bias=True).cuda()
        return w_conv(x)

    def hw_mask(self, x, feat_channels):
        hw_mask = nn.Conv2d(feat_channels, feat_channels, 1).cuda()
        return hw_mask(x)

    def strippool(self, x, p):
        featmap_sizes = x.size()[-2:]
        x_c = torch.sum(x, dim=2, keepdim=False) # channel wise
        reg_feat_hw = self.hw_conv(x_c, p)
        reg_feat_pre_h_pool = self.h_pool(reg_feat_hw)
        reg_feat_pre_w_pool = self.w_pool(reg_feat_hw)
        reg_feat_h = F.interpolate(self.h_conv(reg_feat_pre_h_pool, p), featmap_sizes, mode='bilinear', align_corners=True)
        reg_feat_w = F.interpolate(self.w_conv(reg_feat_pre_w_pool, p), featmap_sizes, mode='bilinear', align_corners=True)
        reg_feat_pre_pool_hw = (reg_feat_h + reg_feat_w)
        x = self.hw_mask(reg_feat_pre_pool_hw, p).sigmoid().unsqueeze(2).expand_as(x) * x + x
        return x
    def conv(self, x, feat_channels):
        conv = nn.Conv2d(feat_channels, feat_channels, 1).cuda()
        relu = DyReLU(feat_channels).cuda()
        return relu(conv(x))

    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = self.patch_size(h,w)
        x_p = self.proj(x, patch_size).reshape(b, patch_size, patch_size, c, -1).permute(0, 4, 3, 1, 2)  # batch, k x k x channel, num_patch
        _, P, _, _, _ = x_p.shape
        x_p= self.strippool(x_p, P) + x_p
        x = self.re_proj(x_p.permute(0, 2, 3, 4, 1).reshape(b,-1, P), patch_size,(h, w))
        x = self.conv(x, c)
        return x

class DyDCNv2(nn.Module):
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
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset.contiguous(), mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class BorderBranch(nn.Module):
    def __init__(self, in_channels, border_channels):
        """
        :param in_channels:
        """
        super(BorderBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels),
            nn.ReLU())

        self.ltrb_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels * 4),
            nn.ReLU())

        self.border_align = BorderAlign(pool_size=10)
        self.border_mask = nn.Conv2d(4 *border_channels, 5 * border_channels, 3, padding=1)
        self.border_conv = nn.Sequential(
            nn.Conv2d(
                5 * border_channels,
                in_channels,
                kernel_size=1),
            nn.ReLU())
        #self.c_at = CrissCrossAttention(in_channels*5)

    def forward(self, feature, boxes, wh):
        N, C, H, W = feature.shape

        fm_short = self.cur_point_conv(feature) # feat_point
        feature = self.ltrb_conv(feature) # feat_ltrb

        ltrb_conv = self.border_align(feature, boxes)  # core boder code
        ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        mask = self.border_mask(feature).sigmoid()
        align_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        #align_conv = self.c_at(align_conv)
        align_conv = self.border_conv(align_conv * mask)
        return align_conv

class StripPooling(nn.Module):
    def __init__(self, feat_channels):
        super(StripPooling, self).__init__()
        self.feat_channels = feat_channels
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.h_conv = nn.Conv2d(self.feat_channels, self.feat_channels, (1, 3), 1, (0, 1), bias=True).cuda()
        self.w_conv = nn.Conv2d(self.feat_channels, self.feat_channels, (1, 3), 1, (0, 1), bias=True).cuda()
        self.conv = nn.Conv2d(self.feat_channels, self.feat_channels, 1).cuda()

    def forward(self, x):
        featmap_sizes = x.size()[-2:]
        reg_feat_pre_h_pool = self.h_pool(x)
        reg_feat_pre_w_pool = self.w_pool(x)
        reg_feat_h = F.interpolate(self.h_conv(reg_feat_pre_h_pool), featmap_sizes, mode='bilinear',
                                   align_corners=True)
        reg_feat_w = F.interpolate(self.w_conv(reg_feat_pre_w_pool), featmap_sizes, mode='bilinear',
                                   align_corners=True)
        x = (self.conv(reg_feat_h + reg_feat_w) + x).sigmoid() * x
        return x

@HEADS.register_module()
class BoderATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 pred_kernel_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=True,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fpn_strides = [8, 16, 32, 64, 128]
        super(BoderATSSHead, self).__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        #self.batchdrop = Patch_Drop(256, 0.01)


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])
        self.add_module("border_cls_subnet", BorderBranch(self.feat_channels, 256))
        self.add_module("border_bbox_subnet", BorderBranch(self.feat_channels, 256))
        self.border_cls_score = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, kernel_size=1, stride=1)
        #self.refine_reg = PacConv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1)
        #self.refine_cls = PacConv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1)
        #self.offset_pred = nn.Sequential(
            #nn.Conv2d(self.feat_channels,
             #         self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            #nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1))
       # self.strip_pool_cls = StripPooling(self.feat_channels,norm_layer=nn.BatchNorm2d)
       # self.strip_pool_reg = StripPooling(self.feat_channels, norm_layer=nn.BatchNorm2d)
        self.border_bbox_pred= nn.Conv2d(self.feat_channels, self.num_base_priors * 4, kernel_size=3, padding=1)
        self.p_pool = StripPooling(self.feat_channels)


    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
       # for m in self.offset_pred:
         #   if isinstance(m, nn.Conv2d):
         #       normal_init(m, std=0.01)
        if isinstance(self.border_cls_score, nn.Conv2d):
            normal_init(m, std=0.01)
        normal_init(self.border_cls_score, std=0.01, bias=bias_cls)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
       # self.reg_dps = PacConv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=pred_pad_size)
       # self.cls_dps = PacConv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=pred_pad_size)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs= self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list


    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        levels = [0, 1, 2, 3, 4]
        return multi_apply(self.forward_single, feats, self.scales, levels)

    def forward_single(self, x, scale, level=None):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        #if self.training:
        #    _, _,H,W= x.shape
        #    x = self.batchdrop(x, H, W)
        x = self.p_pool(x)
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        #print(cls_score)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        cls_score = self.atss_cls(cls_feat)
        #
        N, C, H, W = x.shape
        shift_p = self.prior_generator.single_level_grid_priors(
            (H, W), level, device=x.device)
        shift_p = torch.cat([shift_p.unsqueeze(0) for _ in range(N)])
        pre_off = bbox_pred.clone().detach()
        with torch.no_grad():
            pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4)
            pre_boxes = self.compute_bbox(shift_p, pre_off)
            align_boxes, wh = self.compute_border(pre_boxes, level, H, W)
        reg_feat_boder = self.border_bbox_subnet(reg_feat, align_boxes, wh)
        #reg_feat_boder = self.refine_reg(reg_feat_boder, x)
        bbox_pred_boder = self.border_bbox_pred(reg_feat_boder+x)

        """bbox_pred decode
        offset = self.offset_pred(reg_feat)
        reg_dist = bbox_pred_boder.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_bbox = self.bbox_coder.decode(shift_p.reshape(-1, 4), reg_dist).reshape(
            N, H, W, 4).permute(0, 3, 1, 2) / self.fpn_strides[level]

        bbox_pred_boder = self.deform_sampling(reg_bbox.contiguous(), offset.contiguous())
        invalid_bbox_idx = (bbox_pred_boder[:, [0]] > bbox_pred_boder[:, [2]]) | \
                           (bbox_pred_boder[:, [1]] > bbox_pred_boder[:, [3]])
        invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred_boder)
        bbox_pred_boder = torch.where(invalid_bbox_idx, reg_bbox, bbox_pred_boder)
        
        """
        cls_feat_boder = self.border_cls_subnet(cls_feat, align_boxes, wh)
        #cls_feat_boder = self.refine_cls(cls_feat_boder, x)
        border_cls_logits = self.border_cls_score(cls_feat_boder+x)
        #border_cls_logits = sigmoid_geometric_mean(cls_score, border_cls_logits)
        centerness = self.atss_centerness(reg_feat_boder)
        if self.training:
            return cls_score, bbox_pred, centerness, bbox_pred_boder, border_cls_logits
        else:
            return border_cls_logits, bbox_pred_boder, centerness#, reg_feat_boder+x
            #return cls_feat_boder#, reg_feat_boder, cls_feat, reg_feat

    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def compute_border(self, _boxes, fm_i, height, width):
        """
        :param _boxes:
        :param fm_i:
        :param height:
        :param width:
        :return:
        """
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:, :, 0].clamp_(min=0, max=width - 1)
        boxes[:, :, 1].clamp_(min=0, max=height - 1)
        boxes[:, :, 2].clamp_(min=0, max=width - 1)
        boxes[:, :, 3].clamp_(min=0, max=height - 1)

        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        return boxes, wh

    def compute_bbox(self, location, pred_offset):
        detections = torch.stack([
            location[:, :, 0] - pred_offset[:, :, 0],
            location[:, :, 1] - pred_offset[:, :, 1],
            location[:, :, 0] + pred_offset[:, :, 2],
            location[:, :, 1] + pred_offset[:, :, 3]], dim=2)

        return detections

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, bbox_pred_boder, border_cls_logits, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        border_cls_logits = border_cls_logits.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_boder = bbox_pred_boder.permute(0, 2, 3, 1).reshape(-1, 4)

        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        loss_cls_boder = self.loss_cls(
            border_cls_logits, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]
            pos_bbox_pred_boder = bbox_pred_boder[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_pred_boder = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred_boder)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            loss_bbox_boder = self.loss_bbox(
                pos_decode_bbox_pred_boder,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_bbox_boder = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

       # beta_c = (1 + torch.exp(-loss_bbox))
        #beta_r = (1 + torch.exp(-loss_cls))

        return loss_cls * 2.0, loss_bbox * 2.0, loss_centerness, centerness_targets.sum(), loss_bbox_boder * 0.5, loss_cls_boder * 0.5

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             bbox_preds_pred,
             border_cls_logits,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor, loss_bbox_boder, loss_cls_boder = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                bbox_preds_pred,
                border_cls_logits,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        loss_bbox_boder = list(map(lambda x: x / bbox_avg_factor, loss_bbox_boder))

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            loss_bbox_boder=loss_bbox_boder,
            loss_cls_boder=loss_cls_boder)

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
