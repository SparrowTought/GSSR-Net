import torch
from models.CBAM import MS_CAM, DepthWiseConv
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
# 26.63.146.163:8211
import torch.nn.functional as F
import pywt
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from  .torch_vertex import Grapher
import  seaborn as sns
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return avgout + maxout

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        out = a_w * a_h

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class WeightCSP(nn.Module):
    def __init__(self, dim):
        super(WeightCSP, self).__init__()

        self.Conv1x1 = nn.Conv2d(dim, dim, 1)
        self.Conv_cat = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.act = nn.Sigmoid()
        self.conv1 = nn.Sequential(nn.Conv2d(dim , dim, 3, 1, 1),nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim),
                                     nn.ReLU(inplace=True))
        self.num_images = 0



    def forward(self, t1, t2):
        B,C,H,W = t1.shape
        outputs = []
        a1 = t1
        a2 = t2
        cosine = self.act(1 - torch.cosine_similarity(t1, t2, dim=1)).unsqueeze(1)
        t1 = t1 * cosine + t1
        t2 = t2 * cosine + t2
        sub = torch.abs(t1 - t2)
        cat = self.Conv_cat(torch.cat((t1, t2), 1))
        out = cat + sub
        out = self.conv1(out) + out
        out = self.conv2(out) + out

        return out

    
class changede(nn.Module):
    def __init__(self,):
        super(changede, self).__init__()
        self.sig =  nn.Sigmoid()

        self.num_images = 0

        self.eps = 1e-10
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        std = torch.std(x, dim=1, keepdim=True)
        # std1, std2 = std.chunk(2, dim=1)
        mean1 = torch.mean(x1, dim=1, keepdim=True)
        filtered_s1 = torch.where(x1 > mean1, x1, torch.zeros_like(x1))
        nonzero_count_s1 = torch.where(filtered_s1 > 0, torch.ones_like(filtered_s1), torch.zeros_like(filtered_s1)).sum(
            dim=1, keepdim=True).float()
        nonzero_count_s1 = torch.where(nonzero_count_s1 == 0, torch.ones_like(nonzero_count_s1), nonzero_count_s1)
        nonzero_count_s1 = filtered_s1.sum(dim=1, keepdim=True) / nonzero_count_s1
        x1 = (x1 - nonzero_count_s1) / (std + self.eps)

        mean2 = torch.mean(x2, dim=1, keepdim=True)
        filtered_s2 = torch.where(x2 > mean2, x2, torch.zeros_like(x2))
        nonzero_count_s2 = torch.where(filtered_s2 > 0, torch.ones_like(filtered_s2), torch.zeros_like(filtered_s2)).sum(
            dim=1, keepdim=True).float()
        nonzero_count_s2 = torch.where(nonzero_count_s2 == 0, torch.ones_like(nonzero_count_s2), nonzero_count_s2)
        nonzero_count_s2 = filtered_s2.sum(dim=1, keepdim=True) / nonzero_count_s2
        x2 = (x2 - nonzero_count_s2) / (std + self.eps)

        sig_x1 = self.sig(x1)
        sig_x2 = self.sig(x2)

        x1 = sig_x1 * x1 + x1
        x2 = sig_x2 * x2 + x2
        return x1, x2

class LCFE(nn.Module):
    def __init__(self, in_d):
        super(LCFE, self).__init__()
        self.in_d = in_d
        self.out_d = in_d

        self.Wave = Wave(in_d)
        self.UIM = Diff_end(in_d)

    def forward(self, x1, x2):
        eg1, eg2 = self.Wave(x1, x2)
        x = self.UIM(eg1, eg2)

        return x



class Wave(nn.Module):
    def __init__(self, vit_dim):
        super(Wave, self).__init__()

        self.WaveEdge = SelectiveWaveEdge(3, 1, False, vit_dim, vit_dim, first=False)

    def forward(self, x, y):
        output1, output2 = self.WaveEdge(x, y)

        return output1, output2



class Diff(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.conv1_x = nn.Conv2d(dim,dim, 1)
        self.conv1_y = nn.Conv2d(dim,dim, 1)

        self.csp = WeightCSP(dim)

        self.groups = dim

        self.num_images =0

    def forward(self,  t1,t2):
        output= []
        B, C, H, W = t1.shape

        t11 = self.conv1_x(t1)
        t22 = self.conv1_y(t2)
        cat = self.csp(t11, t22)


        return cat
class SelectiveWaveEdge(nn.Module):
    def __init__(self, kernel_size, padding, bias, in_channels, out_channels, first=False):
        super(SelectiveWaveEdge, self).__init__()
        self.first = first
        self.in_channels = in_channels
        middle_channels = in_channels // 2
        self.Edge = EdgeAttention(in_channels)
        if self.in_channels == 64:
            n = 4096
        elif self.in_channels == 96:
            n = 1024
        elif self.in_channels == 192:
            n = 256
        elif self.in_channels == 384:
            n = 64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))

        self.Change = changede()

        self.offset_conv = nn.Sequential(nn.Conv2d(in_channels, 2, kernel_size=1), nn.Tanh())

        self.Grap1x = Grapher(in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                              bias=True, stochastic=False, epsilon=0.0, r=1, n=n, drop_path=0.0, relative_pos=True)
        self.Grap2x = Grapher(in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                              bias=True, stochastic=False, epsilon=0.0, r=1, n=n, drop_path=0.0, relative_pos=True)

        
        self.convx_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, 1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True)
                                     )
       
        self.convy_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, 1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True)
                                     )
        self.num_images = 0

    def sample_with_offset(self, x, offset):
        B, C, H, W = x.shape

        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.view(B, -1, H, W).view(
            B, 2, -1, H, W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        sampled_x = F.grid_sample(x, coords, mode='bilinear', padding_mode='border', align_corners=True)
        return sampled_x

    def forward(self, x, y):

        B,C,H,W = x.shape
        
        x, y = self.Edge(x, y)

        features_T1 = self.Grap1x(x)
        features_T2 = self.Grap2x(y)
        
        x_l = self.convx_2(x)
        y_l = self.convy_2(y)

        offset_T1 = self.offset_conv(x_l)
        offset_T2 = self.offset_conv(y_l)

        x_g = self.sample_with_offset(features_T1, offset_T1)
        y_g = self.sample_with_offset(features_T2, offset_T2)
        fuse_x = torch.cat((x_l, x_g), 1)
        fuse_x = self.conv1(fuse_x)

        fuse_y = torch.cat((y_l, y_g), 1)
        fuse_y = self.conv2(fuse_y)

        outx = fuse_x + x
        outy = fuse_y + y


        return outx, outy

class H_qe(nn.Module):
    def __init__(self, in_channels):
        super(H_qe, self).__init__()
        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.sobel_yx = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.sig = nn.Sigmoid()
        self.in_channels = in_channels
        self.sliu = nn.SiLU(inplace=True)
        self.num_images = 0
        self.initialize_weights()

    def forward(self, x, y_HL, y_LH, y_HH):
        B,C,H,W = x.shape
        outputs=[]
        res_HL = y_HH

        y_HL = self.sig(torch.abs(self.edge_x(x))) * y_HL + y_HL
        y_LH = self.sig(torch.abs(self.edge_y(x))) * y_LH + y_LH
        y_HH = self.sig(torch.abs(self.edge_yx(x))) * y_HH + y_HH

        return  y_HL, y_LH, y_HH

    def initialize_weights(self):
            """Initialize the weights of the convolutional layers with Sobel kernels."""

            edge_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            edge_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
            edge_kernel_yx = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
            edge_kernel_x = edge_kernel_x.repeat(self.in_channels, 1, 1, 1)
            edge_kernel_y = edge_kernel_y.repeat(self.in_channels, 1, 1, 1)
            edge_kernel_yx = edge_kernel_yx.repeat(self.in_channels, 1, 1, 1)
            with torch.no_grad():
                self.edge_x.weight = nn.Parameter(edge_kernel_x, requires_grad=False)
                self.edge_y.weight = nn.Parameter(edge_kernel_y, requires_grad=False)
                self.edge_yx.weight = nn.Parameter(edge_kernel_yx, requires_grad=False)


class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.in_channels = in_channels
        self.wt1 = DWTForward(J=1, mode='zero', wave='haar')
        self.wt2 = DWTForward(J=1, mode='zero', wave='haar')
        self.dwt1 = DWTInverse(wave='haar')
        self.dwt2 = DWTInverse(wave='haar')
        # Sobel filter for edge detection in the X direction
        self.H_qe1 = H_qe(in_channels)
        self.H_qe2 = H_qe(in_channels)
        self.conv31 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.Change = changede()

        self.SiLU = nn.SiLU(inplace=False)
        self.num_images = 0

    def forward(self, x,y):
        outputs=[]
        B,C,H,W=x.shape
        res_x = x
        res_y = y


        xL, xH = self.wt1(x)
        yL, yH = self.wt2(y)

        x_HL = xH[0][:, :, 0, :, :]
        x_LH = xH[0][:, :, 1, :, :]
        x_HH = xH[0][:, :, 2, :, :]

        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        xL, yL = self.Change(xL, yL)
        x = self.conv31(x)
        y = self.conv32(y)

        x_HL, x_LH, x_HH = self.H_qe1(x, x_HL, x_LH, x_HH)
        y_HL, y_LH, y_HH = self.H_qe2(y, y_HL, y_LH, y_HH)


        xH_combined = torch.stack([x_HL, x_LH, x_HH], dim=2)
        yH_combined = torch.stack([y_HL, y_LH, y_HH], dim=2)
 
        x_reconstructed = self.dwt1((xL, [xH_combined]))
        y_reconstructed = self.dwt2((yL, [yH_combined]))

        out1 = x_reconstructed + res_x
        out2 = y_reconstructed + res_y

        return out1, out2


class Diff_end(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()


        self.Diff = Diff(dim)

        self.num_images = 0

    def forward(self, t1, t2):
        output = []
        batch_size, C, height, width = t1.shape

        out = self.Diff(t1, t2)

        return out


class Decode(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_redu = nn.Sequential(nn.Conv2d(dim * 2, dim, 1),
                                       nn.BatchNorm2d(dim, dim),
                                       nn.ReLU(inplace=True))


    def forward(self, x, y):

        output = torch.cat([x, y], dim=1)
        output = self.conv_redu(output)



        return output




