import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from mmdet.models.utils import SELayer, DyReLU

class BatchDrop(nn.Module):
    def __init__(self, h_ratio=0.05, w_ratio=0.05):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = int(self.h_ratio * h)
            rw = int(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

class Patch_Drop(nn.Module):
    def __init__(self, inchannels, patch_ratio):
        super(Patch_Drop, self).__init__()
        self.patch_ratio= patch_ratio
        self.in_channels = inchannels
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.15)
    def proj(self,x, patch_size):
        proj = F.unfold(x, kernel_size=patch_size, stride=patch_size).cuda()
        return proj
    def re_proj(self,x, patch_size, output_size):
        re_proj = F.fold(x, output_size=output_size,kernel_size=patch_size, stride=patch_size).cuda()
        return re_proj
    def forward(self,x, H, W):
        b, c, _, _= x.shape
        patch_size = min(int(H*self.patch_ratio)+1, int(W*self.patch_ratio)+1)
        x_p = self.proj(x, patch_size).reshape(b, patch_size*patch_size, c, -1).permute(0, 3, 2, 1) # batch, k x k x channel, num_patch
        _,p,_,_= x_p.shape
        x_g = self.gap(x).permute(0, 2, 1, 3)
        channel_wise=None
        if channel_wise is not None:
            x_g = torch.sum(x_g, dim=2, keepdim=True)
            x_p = torch.sum(x_p, dim=2, keepdim=True)
        x_g = x_g.expand_as(x_p)
        idx_list=[]
        mask = torch.ones_like(x_p)
        for i, j in zip(x_g, x_p):
            cosin_similarity = torch.cosine_similarity(i.reshape(p, -1), j.reshape(p, -1), dim=-1)
            k = int(len(cosin_similarity) * 0.4) + 1
            _, idx = torch.topk(cosin_similarity, k=k, dim=-1)
            #l = len(cosin_similarity)
            #idx = torch.randperm(l)[:int(l*0.01)+1]
            #_, ids = torch.topk(-cosin_similarity, k=1, dim=-1)
            #ids_list.append(ids)
            idx_list.append(idx)
        for k in range(len(idx_list)):
            mask[k, idx_list[k], :, :] = self.dropout(mask[k, idx_list[k], :, :])
        mask = mask.reshape(b, p, -1).permute(0,2,1)
        mask = self.re_proj(mask, patch_size,(H,W))
        x = mask * x
        return x


class Patch_Easre(nn.Module):
    def __init__(self, inchannels, embed_dim, patch_ratio):
        super(Patch_Easre, self).__init__()
        self.patch_ratio= patch_ratio
        self.in_channels = inchannels
        self.embed_dim = embed_dim
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    def proj(self,x, patch_size):
        proj = F.unfold(x, kernel_size=patch_size, stride=patch_size).cuda()
        return proj
    def re_proj(self,x, patch_size, output_size):
        re_proj = F.fold(x, output_size=output_size,kernel_size=patch_size, stride=patch_size).cuda()
        return re_proj
    def forward(self,x, H, W):
        b, c, _, _= x.shape
        patch_size = min(int(H*self.patch_ratio)+1, int(W*self.patch_ratio)+1)
        x_p = self.proj(x, patch_size).reshape(b, patch_size*patch_size, c, -1).permute(0, 3, 2, 1) # batch, k x k x channel, num_patch
        _,p,_,_= x_p.shape
        x_g = self.gap(x).permute(0, 2, 1, 3)
        channel_wise=None
        if channel_wise is not None:
            x_g = torch.sum(x_g, dim=2, keepdim=True)
            x_p = torch.sum(x_p, dim=2, keepdim=True)
        x_g = x_g.expand_as(x_p)
        idx_list=[]
        ids_list=[]
        for i, j in zip(x_g, x_p):
            cosin_similarity = torch.cosine_similarity(i.reshape(p, -1), j.reshape(p, -1), dim=-1)
            _, idx = torch.topk(cosin_similarity, k=1, dim=-1)
            _, ids = torch.topk(-cosin_similarity, k=1, dim=-1)
            ids_list.append(ids)
            idx_list.append(idx)
        for k in range(len(idx_list)):
            x_p[k, idx_list[k], :, :] = x_p[k, ids_list[k], :, :]
        x_p = x_p.reshape(b, p, -1).permute(0,2,1)
        mask = self.re_proj(x_p, patch_size,(H,W))
        x = mask * x
        return x

class FeatGuide_Drop(nn.Module):
    def __init__(self, in_channels, base_ratio=0.01):
        super(FeatGuide_Drop, self).__init__()
        self.se = SELayer(in_channels).to('cuda')
        self.w_i = nn.Parameter(torch.zeros(1))
        self.h_ratio = base_ratio * 1 # area<0.04 -> 0.86,  area<0.03 -> 0.82
        self.w_ratio = base_ratio * 2

    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            #x = self.se(x) # bchw
            f_x = F.unfold(x, kernel_size=3, padding=1) # b cx9, num_patch
            f_x = f_x.permute(0, 2, 1).reshape(b, -1, c, 9)
            f_x = F.adaptive_avg_pool2d(f_x, (1,1)) #
            f_x = f_x.reshape(b, 1, h, w).sigmoid()
            _, sy_s = torch.max(f_x, dim=-2)
            _, sx_s = torch.max(f_x, dim=-1)
            rh = int(self.h_ratio * h)
            rw = int(self.w_ratio * w)
            sx_i = torch.randint(0, h-rh, (b,1,1))
            sy_i = torch.randint(0, w-rw, (b,1,1))
           # mask_sx = sx_s.new_zeros(sx_s.size())
            #print(mask_sx.shape,mask_sx)
            mask = x.new_ones(x.size())
            for sx_, sy_, sx_ii, sy_ii, mask_i in zip(sx_s, sy_s, sx_i.squeeze(-1), sy_i.squeeze(-1), mask):
                sx=(sx_[:,sx_ii[0]])
                sy=(sy_[:, sy_ii[0]])
                mask_i[:, sx:sx + rh, sy:sy + rw] = 0
            #sx = sx_s[sx_i]
            #sy = sy_s[sy_i]

           # print(sx, sy, sx.shape, sy.shape)

            #mask_thr = self.w_i * torch.var(f_x.reshape(b, 1, -1), dim=-1, keepdim=True) + torch.mean(f_x.reshape(b, 1, -1), dim=-1, keepdim=True)
            #mask_thr = mask_thr.unsqueeze(-1).expand_as(f_x)
            #mask = f_x.new_ones(f_x.size())
           # print(mask, mask.shape)
            #mask[mask>mask_thr] = 0
           # print(mask, mask.shape)
            x = x * mask
        return x

class FeatGuide_BatchDrop(nn.Module):
    def __init__(self, in_channels, h_ratio=0.05, w_ratio=0.1):
        super(FeatGuide_BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1).to('cuda')
        self.dyrelu_h = DyReLU(in_channels).to('cuda')
        self.conv_w = nn.Conv2d(in_channels, in_channels, 1).to('cuda')
        self.dyrelu_w = DyReLU(in_channels).to('cuda')
        self.se = SELayer(in_channels).to('cuda')

    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            x = self.se(x) # bchw
            g_x = F.adaptive_avg_pool2d(x, (1,1)) #bc11

            sx = self.conv_h(g_x)
            sx = self.dyrelu_h(sx).sigmoid()

            sy = self.conv_w(g_x)
            sy = self.dyrelu_w(sy).sigmoid()

            sx = h * sx
            sy = h * sy
            sx = sx.ceil()
            sy = sy.ceil() # b c 1 1

            rh = int(self.h_ratio * h)
            rw = int(self.w_ratio * w)

            sx[sx > h - rh] = h - rh
            sy[sy > w - rw] = w - rw

            mask = x.new_ones(x.size()) # bchw
            for sx_i, sy_i, m_i in zip(sx, sy, mask):
                sx_idx = sx_i.squeeze(-1).squeeze(-1).tolist()
                sy_idx = sy_i.squeeze(-1).squeeze(-1).tolist()
                for m, j, k in zip(m_i, sx_idx, sy_idx):
                    m[int(j):int(j) + rh, int(k):int(k) + rw] = 0
            #mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x
