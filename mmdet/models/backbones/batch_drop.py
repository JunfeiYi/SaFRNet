import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from ..utils import SELayer, DyReLU

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
class FeatGuide_Drop(nn.Module):
    def __init__(self, in_channels):
        super(FeatGuide_Drop, self).__init__()
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1).to('cuda')
        self.dyrelu_h = DyReLU(in_channels).to('cuda')
        self.conv_w = nn.Conv2d(in_channels, in_channels, 1).to('cuda')
        self.dyrelu_w = DyReLU(in_channels).to('cuda')
        self.se = SELayer(in_channels).to('cuda')
        self.w_i = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            x = self.se(x) # bchw
            f_x = F.unfold(x, kernel_size=3, padding=1) # b cx9, num_patch
            f_x = f_x.permute(0, 2, 1).reshape(b, -1, c, 9)
            f_x = F.adaptive_avg_pool2d(f_x, (1,1)) #
            f_x = f_x.reshape(b, 1, h, w).sigmoid()
            mask_thr = self.w_i * torch.var(f_x.reshape(b, 1, -1), dim=-1, keepdim=True) + torch.mean(f_x.reshape(b, 1, -1), dim=-1, keepdim=True)
            mask_thr = mask_thr.unsqueeze(-1).expand_as(f_x)
            mask = f_x.new_ones(f_x.size())
           # print(mask, mask.shape)
            mask[mask>mask_thr] = 0
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
