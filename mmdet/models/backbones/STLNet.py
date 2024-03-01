from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import module


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class QCO_1d(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), padding=0, stride=1)
        self.kernel_size = 3

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape # 25, 42
        pat_x_0 = self.unfold(x) # batch, c X kernel x kernel , num_patch
        pat_x = pat_x_0.permute(0,2,1)
        _, P, _= pat_x.shape
        pat_x = pat_x.reshape(N, P, C, self.kernel_size, -1)
        pat_x = pat_x.permute(0, 2, 1, 3, 4)
        pat_avg = F.adaptive_avg_pool3d(pat_x, (P, 1, 1))
       # cos_sim_0 = (F.normalize(pat_x, dim=1) * F.normalize(pat_avg, dim=1)).sum(1).unsqueeze(1).reshape(N, P, -1)
        pat_x = pat_x.sum(1)
        pat_avg = pat_avg.sum(1)
       # cos_sim = (F.normalize(pat_x, dim=1) * F.normalize(pat_avg, dim=1)).unsqueeze(1).reshape(N, P, -1)
        cos_sim = (F.normalize(pat_x.reshape(N, P, -1), dim=1) * F.normalize(pat_avg.reshape(N, P, -1), dim=1))
       # cos_sim = (F.normalize(pat_x, dim=1) * F.normalize(pat_avg, dim=1)).sum(1).unsqueeze(1)
        #cos_sim = cos_sim.reshape(N, P, self.kernel_size, -1)
        cos_sim = cos_sim.permute(0, 2, 1)
        self.fold = torch.nn.Fold(output_size=(H,W), kernel_size=(self.kernel_size, self.kernel_size), padding=0, stride=1)
        cos_sim = self.fold(cos_sim).squeeze(1)

        #'''
        cos_sim = cos_sim.unsqueeze(1)
        unfold_s = torch.nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=1, stride=1)  # batch, c X kernel x kernel , num_patch
        cos_sim_uf = unfold_s(cos_sim)
        b_s, _,p_s= cos_sim_uf.shape
        cos_sim_uf = cos_sim_uf.permute(0,2,1)
        cos_sim_uf = cos_sim_uf.reshape(b_s, p_s, self.kernel_size, -1) #kernel_size
        cos_sim = F.adaptive_max_pool2d(cos_sim_uf, (1, 1))
        #'''
        cos_sim = cos_sim.reshape(N, -1)
        cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = torch.arange(self.level_num).float().cuda()  # 128 : [0, 127]
        q_levels = q_levels.expand(N, self.level_num)  # N x 127
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        quant = 1 - torch.abs(q_levels - cos_sim)  # abs:绝对值
        quant = quant * (quant > (1 - q_levels_inter))
        sta = quant.sum(1)
        sta = sta / (sta.sum(-1).unsqueeze(-1))
        sta = sta.unsqueeze(1)
        sta = torch.cat([q_levels, sta], dim=1)
        sta = self.f1(sta)
        sta = self.f2(sta) # 4 128 128

        x_ave = pat_x_0.permute(0, 2, 1)
        x_ave = x_ave.reshape(N, P, C, -1)
        x_ave = x_ave.permute(0, 2, 1, 3)
        x_ave = F.adaptive_avg_pool2d(x_ave, (1, 1)) # 4, 128,1,1
        x_ave = x_ave.squeeze(-1).squeeze(-1) #4, 128
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0) #4, 128 128  N=4
        sta = torch.cat([sta, x_ave], dim=1)
        sta = self.out(sta)
        return sta, quant


class QCO_1d_ch(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d_ch, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape  # 25, 42
        x_ave = F.adaptive_avg_pool2d(x, (1, 1))  # combine Channel and patch
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1)

        cos_sim = cos_sim.contiguous().view(N, -1)
        cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = torch.arange(self.level_num).float().cuda()   # 128 : [0, 127]
        q_levels = q_levels.expand(N, self.level_num)  # N x 127
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        quant = 1 - torch.abs(q_levels - cos_sim)  # abs:绝对值
        quant = quant * (quant > (1 - q_levels_inter))
        sta = quant.sum(1)
        sta = sta / (sta.sum(-1).unsqueeze(-1))
        sta = sta.unsqueeze(1)
        sta = torch.cat([q_levels, sta], dim=1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        x_ave = x_ave.squeeze(-1).squeeze(-1)
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0)
        sta = torch.cat([sta, x_ave], dim=1)
        sta = self.out(sta)
        return sta, quant

'''
class QCO_2d(nn.Module):
    def __init__(self, scale, level_num): #1 8
        super(QCO_2d, self).__init__()
        self.f1 = nn.Sequential(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.Sequential(ConvBNReLU(256+128, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'), ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num
    def forward(self, x):
        N1, C1, H1, W1 = x.shape # pre_shape
        if H1 // self.level_num != 0 or W1 // self.level_num != 0: # H1,W1< level_num
            x = F.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale)) 
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W))
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1) 
        cos_sim = cos_sim.unsqueeze(1) 
        cos_sim = cos_sim.reshape(N, 1, self.scale, self.size_h, self.scale, self.size_w)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.scale*self.scale), int(self.size_h*self.size_w)) 
        cos_sim = cos_sim.permute(0, 1, 3, 2) 
        cos_sim = cos_sim.squeeze(1)
        cos_sim_min, _ = cos_sim.min(1) 
        cos_sim_min = cos_sim_min.unsqueeze(-1) 
        cos_sim_max, _ = cos_sim.max(1) 
        cos_sim_max = cos_sim_max.unsqueeze(-1) 
        q_levels = torch.arange(self.level_num).float().cuda() 
        q_levels = q_levels.expand(N, self.scale*self.scale, self.level_num) 
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min 
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] 
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1) 
        cos_sim = cos_sim.unsqueeze(-1) 
        q_levels = q_levels.unsqueeze(1)
        quant = 1 - torch.abs(q_levels - cos_sim) 
        quant = quant * (quant > (1 - q_levels_inter)) 
        quant = quant.contiguous().view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num]) 
        quant = quant.permute(0, -2, -1, 1, 2) 
        quant = quant.contiguous().view(N, -1, self.size_h, self.size_w)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.contiguous().view(N, self.scale*self.scale, self.level_num, self.size_h+1, self.size_w+1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3) 
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2) 
        quant = quant_left * quant_right 
        sta = quant.sum(-1).sum(-1) 
        sta = sta / (sta.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6) 
        sta = sta.unsqueeze(1) 
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale*self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4) 
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3) 
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1) 
        sta = sta.contiguous().view(N, 3, self.scale * self.scale, -1) 
        sta = self.f1(sta)
        sta = self.f2(sta) 
        x_ave = x_ave.contiguous().view(N, C, -1) 
        x_ave = x_ave.expand(self.level_num*self.level_num, N, C, self.scale*self.scale)
        x_ave = x_ave.permute(1, 2, 3, 0) 
        sta = torch.cat([x_ave, sta], dim=1)
        sta = self.out(sta) 
        sta = sta.mean(-1)
        sta = sta.contiguous().view(N, sta.shape[1], self.scale, self.scale)
        return sta
'''

class TEM(nn.Module):
    def __init__(self, level_num):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.qco_ch = QCO_1d_ch(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.k_c = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q_c = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v_c = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out_c = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')
        self.dc = ConvBNReLU(512, 256, 1, 1, 0)

    def forward(self, x):

        N, C, H, W = x.shape

        sta_c, quant_c = self.qco_ch(x)
        k_c = self.k_c(sta_c)
        q_c = self.q_c(sta_c)
        v_c = self.v_c(sta_c)
        k_c = k_c.permute(0, 2, 1)
        w_c = torch.bmm(k_c, q_c)
        w_c = F.softmax(w_c, dim=-1)
        v_c = v_c.permute(0, 2, 1)
        f_c = torch.bmm(w_c, v_c)
        f_c = f_c.permute(0, 2, 1)
        f_c = self.out_c(f_c)
        quant_c = quant_c.permute(0, 2, 1)
        out_c = torch.bmm(f_c, quant_c)
        out_c = out_c.contiguous().view(N, 256, H, W)

        sta, quant = self.qco(x)
        k = self.k(sta)
        q = self.q(sta) 
        v = self.v(sta)
        k = k.permute(0, 2, 1) 
        w = torch.bmm(k, q) 
        w = F.softmax(w, dim=-1)
        v = v.permute(0, 2, 1)
        f = torch.bmm(w, v) 
        f = f.permute(0, 2, 1)
        f = self.out(f)
        quant = quant.permute(0, 2, 1)
        out = torch.bmm(f, quant)
        out = out.contiguous().view(N, 256, H, W)
       # out = torch.cat((out, out_c), dim=1)
        out = torch.cat((out_c, out), dim=1)
       # out = self.dc(out)

        return out


'''
class PTFEM(nn.Module):
    def __init__(self):
        super(PTFEM, self).__init__()
        self.conv = ConvBNReLU(512, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.qco_1 = QCO_2d(1, 8)
        self.qco_2 = QCO_2d(2, 8)
        self.qco_3 = QCO_2d(3, 8)
        self.qco_6 = QCO_2d(6, 8)
        self.out = ConvBNReLU(512, 256, 1, 1, 0)
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.conv(x) # C: 512 -> 256 
        sta_1 = self.qco_1(x) 
        sta_2 = self.qco_2(x)
        sta_3 = self.qco_3(x)
        sta_6 = self.qco_6(x)   
        N, C = sta_1.shape[:2]
        sta_1 = sta_1.contiguous().view(N, C, 1, 1)
        sta_2 = sta_2.contiguous().view(N, C, 2, 2)
        sta_3 = sta_3.contiguous().view(N, C, 3, 3)
        sta_6 = sta_6.contiguous().view(N, C, 6, 6)
        sta_1 = F.interpolate(sta_1, size=(H, W), mode='bilinear', align_corners=True)
        sta_2 = F.interpolate(sta_2, size=(H, W), mode='bilinear', align_corners=True)
        sta_3 = F.interpolate(sta_3, size=(H, W), mode='bilinear', align_corners=True)
        sta_6 = F.interpolate(sta_6, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([sta_1, sta_2, sta_3, sta_6], dim=1) 
        x = self.out(x)
        return x


'''

class STL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_upsam = ConvBNReLU(512, 1024, 3, 2, 1)
        self.conv_downch = ConvBNReLU(2048, 256, 1, 1, 0)
       # self.down_sample = ConvBNReLU(256, 512, 3, 2, 1)
       # self.conv_start = ConvBNReLU(1024, 256, 1, 1, 0)
        self.tem = TEM(128)
        self.conv_0 = ConvBNReLU(1536, 1024, 1, 1, 0)
        #self.conv_1 = ConvBNReLU(3328, 1024, 1, 1, 0)
       # self.conv_4 = ConvBNReLU(3072, 2048, 1, 1, 0)
    def forward(self, x0, x1):
        '''
        unfold_1 = torch.nn.Unfold(kernel_size=(5, 5), stride=1)
        x_3 = self.conv_2(x3)
        x4 = self.conv_3(x_3) #torch.Size([4, 256, 25, 42])
        _, _, h,w = x4.shape
        patches = unfold_1(x4) # batch, c X kernel x kernel , num_patch
        patches = patches.permute(0,2,1) # batch, num_patch, c X kernel x kernel
        b, p, _ = patches.shape
        patches_fla = patches.contiguous().view(b, p, -1)
        patches_fla_t = patches.permute(0,2,1)
        gram_m = torch.bmm(patches_fla, patches_fla_t)
        N= h * w
        gram_m = gram_m.reshape(b,-1)
        '''

       # outs=[]
        #outs.append(x0) # 256
        x0 = self.conv_upsam(x0)
        #_, _, H, W = x2.shape
        #x = self.conv_upsam(x3)
        x_fu = torch.cat([x1, x0], dim=1)
        x = self.conv_downch(x_fu) #torch.Size([4, 256, 25, 42])
        x_tem = self.tem(x)

        #x_tem = F.interpolate(x_tem, size=(H, W), mode='bilinear', align_corners=True)
        x_fuse = torch.cat([x1, x_tem], dim=1)
        x1 = self.conv_0(x_fuse)
       # x3_dp = F.interpolate(x_fuse, size=(H, W), mode='bilinear', align_corners=True)
      #  x2 = torch.cat([x2, x3_dp], dim=1)
       # x2 = self.conv_1(x2)
       # outs.append(x1)
       # outs.append(x2)
      #  outs.append(x3)
        return x1


    '''
    def forward(self, x0, x1, x3):   
       # x0 = self.down_sample(x0)
       # x = torch.cat([x0, x1],dim=1)
        _, _, H, W = x0.shape
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1],dim=1)
        x = self.conv_start(x) 
        x_tem = self.tem(x)
        x = torch.cat([x_tem, x], dim=1) #c = 256 + 256 = 512
        x_ptfem = self.ptfem(x) # 256        
        #x_ptfem = self.up_channels(x_ptfem)   
        x = torch.cat([x_ptfem, x], dim=1) # c = 768 = 256 + 512
        x = self.conv1(x)
        x = self.conv2(x)       
        x = torch.cat([x, x3],dim=1)
        x = self.down_channels_1(x)
        
        return x
    '''
