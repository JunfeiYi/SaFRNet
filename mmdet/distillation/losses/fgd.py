import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init, normal_init
from ..builder import DISTILL_LOSSES


class UncertaintyWithLossFeature(nn.Module):
    def __init__(self):
        super(UncertaintyWithLossFeature, self).__init__()
        feat_channels = 32
        self.conv_1 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            )
        self.conv_2 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            )
        self.conv_3 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            )
        self.conv_4 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # yapf: disable
            )      
                                      
        self.iou_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_channels),
            nn.ReLU(),
            nn.Linear(in_features=feat_channels, out_features=feat_channels),
            nn.ReLU()
        )
        self.prob_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_channels),
            nn.ReLU(),
            nn.Linear(in_features=feat_channels, out_features=feat_channels),
            nn.ReLU()
        )
        self.cls_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_channels),
            nn.ReLU(),
            nn.Linear(in_features=feat_channels, out_features=feat_channels),
            nn.ReLU()
        )
        self.reg_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_channels),
            nn.ReLU(),
            nn.Linear(in_features=feat_channels, out_features=feat_channels),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=feat_channels * 4, out_features=feat_channels),
            nn.ReLU(),
            nn.Linear(in_features=feat_channels, out_features=1),
        )
        self.init_weight()

    def init_weight(self):
        for m in [self.iou_net, self.prob_net, self.cls_loss_net, self.reg_loss_net, self.predictor]:
            normal_init(m[0], mean=0.0, std=0.0001, bias=0)
            normal_init(m[2], mean=0.0, std=0.0001, bias=0)

    def forward(self, ious, probs, cls_loss, reg_loss):
      
        ious = self.conv_1(ious)
        probs = self.conv_2(probs)
        cls_loss = self.conv_3(cls_loss)
        reg_loss = self.conv_4(reg_loss)

        iou_feature = self.iou_net(ious.view(-1, 1))
        probs_feature = self.prob_net(probs.reshape(-1, 1))
        cls_loss_feature = self.cls_loss_net(cls_loss.reshape(-1, 1))
        reg_loss_feature = self.reg_loss_net(reg_loss.reshape(-1, 1))
        non_visual_input = torch.cat((iou_feature, probs_feature, cls_loss_feature, reg_loss_feature), dim=1)
        prediction = self.predictor(non_visual_input)
        
        return prediction


@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.uncertainty_predictor = UncertaintyWithLossFeature()

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        
        self.scale_s1 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=1, padding=1)
        self.scale_s2 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=2, padding=2)
        self.scale_s3 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=3, padding=3)
        self.dss = nn.Sequential(
                 nn.Conv2d(teacher_channels*4, teacher_channels, kernel_size=1),
                 nn.BatchNorm2d(teacher_channels, affine=False))
        
        self.scale_t1 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=1, padding=1)
        self.scale_t2 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=2, padding=2)
        self.scale_t3 = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3,
                     dilation=3, padding=3)
        self.dst = nn.Sequential(
                 nn.Conv2d(teacher_channels*4, teacher_channels, kernel_size=1),
                 nn.BatchNorm2d(teacher_channels, affine=False))
        self.reset_parameters()
        #self.p = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

         
    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        mask_loss_c, mask_loss_s = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = torch.sum(mask_loss_c) / mask_loss_c.numel() + torch.sum(mask_loss_s) / mask_loss_s.numel()
        
        rela_loss = self.get_rela_loss(preds_S, preds_T)
        #feat_loss = self.get_feat_loss(preds_S, preds_T)
        #print(feat_loss.shape, rela_loss.shape)
       # mask_loss_c = mask_loss_c.unsqueeze(-1).unsqueeze(-1).expand_as(rela_loss)
       # mask_loss_s = mask_loss_s.unsqueeze(1).expand_as(rela_loss)
        
        """
        if H*W > 2000:
            mask_loss_c_c = mask_loss_c.sum(1) / C
            mask_loss_s_c = mask_loss_s.sum(1) / C
            rela_loss_c = rela_loss.sum(1) / C
            feat_loss_c = feat_loss.sum(1) / C
        """
        #uncertainty_prediction = self.uncertainty_predictor(
       #     mask_loss_c.detach().data,
       #     mask_loss_s.detach().data,
       #     rela_loss.detach().data,
       #     feat_loss.detach().data
       # )
       # uncertainty_prediction_fea = torch.clamp(uncertainty_prediction, min=2.0, max=-2.0)
        
        #uncert_num = uncertainty_prediction_fea.sum() / uncertainty_prediction_fea.numel() * 0.5
        #uncert_pre = uncertainty_prediction_fea
       # uncertainty_prediction_fea = torch.exp(-1. * uncertainty_prediction_fea)
        
       # losses.update({"loss_uncertainty_cls": uncertainty_prediction_cls.sum() / uncertainty_prediction_cls.numel() * self.uncertainty_cls_weight})
        
        #uncertainty_prediction_ = uncertainty_prediction[:, 1]
        #fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, C_attention_s, C_attention_t, S_attention_s, S_attention_t, uncertainty_prediction_fea, uncert_pre)
        
        rela_loss = torch.sum(rela_loss) / rela_loss.numel()

        #print(rela_loss.shape, mask_loss.shape, mask_loss.shape)
        loss =  self.lambda_fgd * rela_loss  +  self.gamma_fgd * mask_loss#  self.beta_fgd * bg_loss + self.alpha_fgd * fg_loss +

        return loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention
        
    def get_feat_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='none')
        feat_loss = loss_mse(preds_S, preds_T)/len(preds_S)
        
        return feat_loss

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t, uncet_fea, uncert_pre):
        loss_mse = nn.MSELoss(reduction='sum')
        b, c, h, w = preds_S.shape
        uncet_fea = uncet_fea.reshape(b, 1, h, w)
        uncert_pre = uncert_pre.reshape(b, 1, h, w)
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        #fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        #fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fea_t = torch.mul(preds_T, torch.sqrt(uncet_fea))
        
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        #fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        #fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fea_s = torch.mul(preds_S, torch.sqrt(uncet_fea))
        
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))
        
        uncert_fg = torch.mul(uncert_pre, Mask_fg)
        uncert_bg = torch.mul(uncert_pre, Mask_bg)
        
        uncert_fg = uncert_fg.sum() / uncert_fg.numel() 
        uncert_bg = uncert_bg.sum() / uncert_bg.numel() 

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg) + uncert_fg * 1.0
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg) + uncert_bg * 1.0

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        loss_mse_c = nn.MSELoss(reduction='none')
        loss_mse_s = nn.MSELoss(reduction='none')
        
        #mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)
        
        mask_loss_c = loss_mse_c(C_s, C_t)/len(C_s) # [4, c]
        mask_loss_s = loss_mse_s(S_s, S_t)/len(S_s) # [4, h, w]
        
        
        return mask_loss_c, mask_loss_s
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='none')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t
        
        
        out_s = torch.cat([self.scale_s1(out_s) , self.scale_s2(out_s) , self.scale_s3(out_s) , preds_S], dim =1)
        out_s = self.dss(out_s)
        out_t = torch.cat([self.scale_t1(out_t) , self.scale_t2(out_t) , self.scale_t3(out_t) , preds_T], dim=1)
        out_t = self.dst(out_t)

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)
