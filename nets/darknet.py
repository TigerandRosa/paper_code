#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F
import warnings
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as function


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
                      dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# CBS-CBF 将其替换基本的baseconv
class FReLU(nn.Module):
    """
    FReLU https://arxiv.org/abs/2007.11824
    """

    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        # 定义漏斗条件T(x)  参数池窗口（Parametric Pooling Window ）来创建空间依赖
        # nn.Con2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True)
        # 使用 深度可分离卷积 DepthWise Separable Conv + BN 实现T(x)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        # f(x)=max(x, T(x))
        return torch.max(x, self.bn(self.conv(x)))


class GCT(nn.Module):

    def __init__(self, learnable=False):
        super(GCT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.learnable = learnable
        if self.learnable:
            self.c = nn.Parameter(torch.ones(1) * 0)
        else:
            self.c = 2

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        attn = self.avg_pool(x).view(b, c)
        # norm
        attn = self.norm(attn)
        # gaussian function
        if self.learnable:
            attn = self.gaussian(attn, 3 * torch.sigmoid(self.c) + 1)
        else:
            attn = self.gaussian(attn, self.c)
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        out = residual * attn
        return out

    @staticmethod
    def norm(x):
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        rst = (x - mean) / std
        return rst

    @staticmethod
    def gaussian(x, c):
        return torch.exp(-(x ** 2) / (2 * c))


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class GAUSS_ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, learnable=False):
        super(GAUSS_ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.learnable = learnable
        if self.learnable:
            self.c = nn.Parameter(torch.ones(1) * 0)
        else:
            self.c = 2

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # norm
        y = self.norm(y)
        # gaussian function
        if self.learnable:
            attn = self.gaussian(y, 3 * torch.sigmoid(self.c) + 1)
        else:
            attn = self.gaussian(y, self.c)

        # Multi-scale information fusion
        y = self.sigmoid(attn)
        return x * y

    @staticmethod
    def norm(x):
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        rst = (x - mean) / std
        return rst

    @staticmethod
    def gaussian(x, c):
        return torch.exp(-(x ** 2) / (2 * c))


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup  # oup：论文Fig2(a)中output的通道数
        init_channels = math.ceil(oup / ratio)  # init_channels: 在论文Fig2(b)中,黄色部分的通道数
        # ceil函数：向上取整，
        # ratio：在论文Fig2(b)中，output通道数与黄色部分通道数的比值
        new_channels = init_channels * (ratio - 1)  # new_channels: 在论文Fig2(b)中，output红色部分的通道数

        self.primary_conv = nn.Sequential(  # 输入所用的普通的卷积运算，生成黄色部分
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            # 1//2=0
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(  # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)  # torch.cat: 在给定维度上对输入的张量序列进行连接操作
        # 将黄色部分和红色部分在通道上进行拼接
        return out[:, :self.oup, :, :]  # 输出Fig2中的output；由于向上取整，可以会导致通道数大于self.out


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x * input


class Res2NetBottleneck(nn.Module):
    expansion = 4  # 残差块的输出通道数=输入通道数*expansion

    # inplanes=32  planes=8  输入是输出的4倍
    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True, norm_layer=True):
        # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if planes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  # BN层
            norm_layer = nn.BatchNorm2d
        # bottleneck_planes = 4*32=128
        bottleneck_planes = groups * planes  # bottleneck_planes = hidden_channels
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        # 1*1的卷积层,在第二个layer时缩小图片尺寸 32-128
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)
        # 3*3的卷积层，一共有3个卷积层和3个BN层 32-32
        self.m = nn.ModuleList([nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,
                                          kernel_size=3, stride=1, padding=1, groups=groups) for _ in
                                range(scales - 1)])
        self.n = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        # 1*1的卷积层，经过这个卷积层之后输出的通道数变成 128-128
        self.conv3 = nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # SE模块
        self.se = SEModule(planes * self.expansion) if se else None

    def forward(self, x):
        identity = x

        # 1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # scales个(3x3)的残差分层架构 128分四块 每块32
        xs = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.n[s - 1](self.m[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.n[s - 1](self.m[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)  # 32-128

        # 1*1的卷积层 128-128
        out = self.conv3(out)
        out = self.bn3(out)

        # 加入SE模块
        if self.se is not None:
            out = self.se(out)
        # 下采样
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(  # 没用到
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv_GS(c1, c2, 3, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP_ECA(nn.Module):
    # VoV-GSCSP https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_GS(c1, c_, 1, 1)
        self.cv2 = Conv_GS(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))
        self.eca = ECA(2 * c_)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = torch.cat((self.m(x1), x1), dim=1)
        x1 = self.eca(x1)
        return self.cv2(x1)


class MP_1(nn.Module):
    def __init__(self, in_channels, k=2):
        super(MP_1, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)
        # 第一分支用于变化通道的1*1卷积
        self.conv1_1 = Conv(in_channels, in_channels, 1, 1)
        # 第二个分支用于变化通道的1*1卷积
        self.conv2_1 = Conv(in_channels, in_channels, 1, 1)
        # 第二个分支用于下采样
        self.conv2_2 = Conv(in_channels, in_channels, 3, 2)

    def forward(self, x):
        # 1分支输出
        x1 = self.conv1_1(self.m(x))
        # 2分支输出
        x2 = self.conv2_2(self.conv2_1(x))
        out = torch.cat([x1, x2], 1)
        return out


class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.in_channels, self.num_codes = in_channels, num_codes
        std = 1. / ((num_codes * in_channels) ** 0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            # 返回填充有未初始化数据的张量,  uniform_--->tensor从均匀分布中抽样数值进行填充 均匀分布就是在该区间内，
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes]scale就是S，比例因子
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    # x, self.codewords, self.scale  scale对应图中的S，比例因子
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        # 在第2维加一个维度 x原先是[batchsize，h*w，Channels] 在第二维加一个维度，然后扩展为
        # expanded_x: [b, h*w, num_codes, in_channels)]
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        # ---处理codebook (num_code, c1)
        # codewords:【num_codes, in_channel】---->
        # [1, 1, num_codes, in_channels]
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))

        # 把scale从1, num_code变成   batch, c2, N, num_codes
        # scale[1,num_code] 变成3维[1, 1, num_codes]
        reshaped_scale = scale.view((1, 1, num_codes))  # N, num_codes

        # ---计算rik = z1 - d  # b, N, num_codes
        # expanded_x & reshaped_codewords的维度不等，但是后两维相等，只要把前两维扩展到与大的相等，就可以
        # dim=3 在第三维做加法 就是num_Codes k
        # 以下算式的计算顺序，先算括号，再算pow(2),再算惩罚，再算sum
        # 跟论文的公式对不上。。。
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()

        # ---处理codebook
        # reshaped_codewords:[1,1,num_Codes,in_channels]
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)

        # ---处理特征向量x b, c1, N
        # 同上 扩展维度
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        # 变换rei  b, N, num_codes,-
        # assignment_weights: [batch_size, channels, num_codes]
        # 在第三维加一个维度
        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,

        # ---开始计算eik,必须在Rei计算完之后
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        # 卷积块的第一个1*1卷积
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        # 这里使用RELU
        self.act1 = act_layer(inplace=True)
        # 第二个3*3卷积
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)
        # 第三个1*1卷积
        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        # 卷积块引入残差连接
        residual = x
        # 1*1卷积
        x = self.conv1(x)
        x = self.bn1(x)
        # 只有drop block不是None的时候才会执行冒号后的语句 不执行
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)  # if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
        # 主分支三个卷积，残差分支一个卷积 然后加一起
        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        # return x_2 是false 走这里 返回残差与主分支之和
        else:
            return x


class LVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_codes=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            # 第一位求平均
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


class LightMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.out_channels = out_channels
        # nn.Groupnorm 将输入的in_channels分为1组 这是设置为1 输入的通道数要能够被组数整除
        self.norm1 = GroupNorm(in_channels)
        self.norm2 = GroupNorm(in_channels)

        # 通道扩张
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        # channel MLP
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        # nn.identity():输入input，输出仍是input
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        # 使用层尺度
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                # torch.ones((out_channels))产生1*out_channels形状的张量，值为1
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        # 使用通道尺度化
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class EVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        # out=256 ch=512
        ch = out_channels * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3,
                               bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(in_channels=in_channels, out_channels=out_channels, num_codes=64)  # c1值暂时未定
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(in_channels, out_channels, ksize=1, stride=1, act="silu", act_layer=nn.GELU,
                                   mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.)
        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # EVC的stem block x1然后分别进入MLP,LVC
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # LVC MLP进行concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        # x最后通道数试256 HW没有缩减
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # the following values are interpreted as false: False, None, numeric zero of all types, and empty strings
        # and containers (including strings, tuples, lists, dictionaries, sets and frozensets). The expression x or y
        # first evaluates x; if x is true, its value is returned; otherwise, y is evaluated and the resulting value
        # is returned.
        # out_feature=NONE,因此本句返回out_feature=in_feature
        out_features = out_features or in_features
        # hidden_feature为True，因此hidden_feature=hidden_feature
        hidden_features = hidden_features or in_features
        # 通道扩张
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        # 恢复通道
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PSA_Channel(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, 1, 1)
        self.cv3 = nn.Conv2d(c_, c1, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.reshape2 = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.layernorm = nn.LayerNorm([c1, 1, 1])

    def forward(self, x):  # shape(batch, channel, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, height*width)
        x2 = self.softmax(self.reshape2(self.cv2(x)))  # shape(batch, height*width)
        y = torch.matmul(x1, x2.unsqueeze(-1)).unsqueeze(-1)  # 高维度下的矩阵乘法（最后两个维度相乘）
        return self.sigmoid(self.layernorm(self.cv3(y))) * x


class PSA_Spatial(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.globalPooling = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # shape(batch, channel, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, height*width)
        x2 = self.softmax(self.globalPooling(self.cv2(x)).squeeze(-1))  # shape(batch, channel/2, 1)
        y = torch.bmm(x2.permute(0, 2, 1), x1)  # shape(batch, 1, height*width)
        return self.sigmoid(y.view(x.shape[0], 1, x.shape[2], x.shape[3])) * x


class PSA(nn.Module):
    def __init__(self, in_channel, parallel=True) -> None:
        super().__init__()
        self.parallel = parallel
        self.channel = PSA_Channel(in_channel)
        self.spatial = PSA_Spatial(in_channel)

    def forward(self, x):
        if (self.parallel):
            return self.channel(x) + self.spatial(x)
        return self.spatial(self.channel(x))


class MixPool(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1)
        self.cv2 = BaseConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)  # 先通过CBL进行通道数的减半
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            z1 = self.n(x)
            z2 = self.n(z1)
            z3 = self.n(z2)
            y1 = y1 + z1
            y2 = y2 + z2
            y3 = y3 + z3
            # 上述两次最大池化
            return self.cv2(torch.cat([x, y1, y2, y3], 1))
            # 将原来的x,一次池化后的y1,两次池化后的y2,3次池化的self.m(y2)先进行拼接，然后再CBL


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        # SPP-maxpool部分
        self.branch1_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.branch2_pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.branch3_pool = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = self.branch1_pool(x1)
        x2 = self.branch2(x)
        x2 = self.branch2_pool(x2)
        x3 = self.branch3(x)
        x3 = self.branch3_pool(x3)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# ------ RFB + SPP_maxpool --------#

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        # MS-CAM的local分支
        self.local_att = nn.Sequential(
            # PW卷积的卷积核为1 *1 * M,M为输入的通道数。第一个卷积缩减通道。
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            # 恢复通道数目
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # global分支
        self.global_att_avg = nn.Sequential(
            # 先把HW都变为1
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att_avg(x)
        y = xl + xg
        y = self.sigmoid(y)
        return y * x + x


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        super().__init__()
        # 为了减少参数我们在两个中间的channel加了个缩放参数r，默认设置为16
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        # 自适应函数的设计空间包含了layer-wise, channel-wise, pixel-wise这三种空间 分别对应的是层, 通道, 像素
        # 这里我们选择了channel-wise 首先分别对H, W维度求均值 然后通过两个卷积层 使得每一个通道所有像素共享一个权重
        # 最后由sigmoid激活函数求得beta
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hardswish":
        module = nn.Hardswish()
    elif name == "frelu":
        module = FReLU()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# Focus和Spd功能一样 只是最后的卷积不一样 Spd降维，Focus升维。
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
        # self.conv = GhostConv(in_channels * 4, out_channels)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


# class BaseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
#         super().__init__()
#         pad         = (ksize - 1) // 2
#         self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
#         self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
#         self.act    = get_activation(act, inplace=True)
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))

# CBS-CBF
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SoftPooling2D(nn.Module):
    def __init__(self, kernel_size, strides=1, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(SoftPooling2D, self).__init__()
        padding = kernel_size // 2
        self.avgpool = nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.cs = Channel_Shuffle(in_ch)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.cs(out)
        out = self.point_conv(out)
        return out


class InvertedResidualsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(InvertedResidualsBlock, self).__init__()
        channels = expansion * in_channels
        self.stride = stride

        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # The shortcut operation does not affect the number of channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.basic_block(x)
        if self.stride == 1:
            out = out + self.shortcut(x)

        return out


activation_table = {'relu': nn.ReLU(),
                    'silu': nn.SiLU(),
                    'hardswish': nn.Hardswish()
                    }


class ConvModule(nn.Module):
    '''A combination of Conv + BN + Activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1,
                 bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class SPPFModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


def autopad_GS(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_GS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad_GS(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSConv(nn.Module):
    def __init__(self, c1, c2, k, s, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv_GS(c1, c_, k, s, None, g, act)  # g:gract：分组卷积
        self.cv2 = Conv_GS(c_, c_, 5, 1, None, c_, act)  # 分组为c_

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class NormalPlusGHostConv(nn.Module):
    def __init__(self, c1, c2, k, s, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv_GS(c1, c_, k, s, None, g, act)  # g:gract：分组卷积 普通卷积
        self.cv2 = GhostConv(c_, c_)  # 分组为c_ DW卷积

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class MP(nn.Module):
    def __init__(self, in_c, k=2):
        super(MP, self).__init__()
        hid_c = in_c // 2
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)
        self.conv1 = BaseConv(in_c, hid_c, 1, 1)  # 第一分支通道数目变化

        self.conv2 = BaseConv(in_c, hid_c, 1, 1)  # 第二分支通道数目变化
        self.conv3 = BaseConv(hid_c, hid_c, 1, 2)

    def forward(self, x):
        x1 = self.conv1(self.m(x))
        x2 = self.conv3(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        return x


class SPPBottleneck_softpool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        # self.conv1      = GhostConv(in_channels, hidden_channels)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)
        # self.conv2      = GhostConv(conv2_channels, out_channels)
        # self.conv2        = DEPTHWISECONV(conv2_channels, out_channels)
        # self.conv2        = NormalPlusGHostConv(conv2_channels, out_channels, 1, 1)

        self.sp_5 = SoftPooling2D(5)
        self.sp_9 = SoftPooling2D(9)
        self.sp_13 = SoftPooling2D(13)

    def forward(self, x):
        x = self.conv1(x)
        x_softpool_5 = self.sp_5(x)
        x_softpool_9 = self.sp_9(x)
        x_softpool_13 = self.sp_13(x)
        x = torch.cat((x, x_softpool_13, x_softpool_9, x_softpool_5), dim=1)
        x = self.conv2(x)
        return x


class SPP_softpool_serial(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=5, activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)
        self.sp_5 = SoftPooling2D(kernel_size=kernel_sizes)

    def forward(self, x):
        x = self.conv1(x)
        x_softpool_5 = self.sp_5(x)
        x_softpool_9 = self.sp_5(x_softpool_5)
        x = self.conv2(torch.cat((x, self.sp_5(x_softpool_9), x_softpool_9, x_softpool_5), dim=1))
        return x


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SPPBottleneck_avgpool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.AvgPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class GhostConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostConv, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=1.0, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class Bottleneck_backbone(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=1.0, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = GhostConv(hidden_channels, out_channels)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class Res2NetBlock(nn.Module):
    def __init__(self, inplanes, outplanes, scales=4):
        super(Res2NetBlock, self).__init__()

        if outplanes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        # 1*1的卷积层 256-32
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, 256, 1, 1, 0),
            nn.BatchNorm2d(256)
        )
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(function.relu(self.conv1(xs[1])))
        ys.append(function.relu(self.conv2(xs[2]) + ys[1]))
        ys.append(function.relu(self.conv2(xs[3]) + ys[2]))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = function.relu(y + input)

        return output


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)
        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)


class Bottleneck_Dark5(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=1.0, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        # self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.conv2 = GhostConv(hidden_channels, out_channels)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer_Dark5(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck_Dark5(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)
        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)


class ConvAWS2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super()._conv_forward(x, weight, None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        weight = self.weight.data
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                                                 keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=autopad(kernel_size, p),
            dilation=d,
            groups=g,
            bias=bias)
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=s,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return self.act(self.bn(out))


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class CSPLayer_neck(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

        self.eca = ECA(2 * hidden_channels)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)
        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # concat后加注意力
        x = self.eca(x)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu", ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act="silu")
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck_softpool(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise,
                     act=act),

        )


    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        # x = self.simam_Dark2(x)
        outputs["dark2"] = x
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        # x = self.simam_Dark3(x)
        outputs["dark3"] = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        # x = self.simam_Dark4(x)
        outputs["dark4"] = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        # x = self.simam_Dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))