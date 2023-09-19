# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: Attention.py
@Author: nkul
@Date: 2023/4/12 下午3:05
Attention Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        return torch.mul(x, y)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([avg_out, max_out], dim=1)
        m = self.conv1(m)
        return x * self.sigmoid(m)


class HiCBAM(nn.Module):
    """
    Input: B * C * H * W
    Out:   B * C * H * W
    """
    def __init__(self, channels) -> None:
        super(HiCBAM, self).__init__()
        self.pixel_attention = PA(channels)
        self.space_attention = SA()

    def forward(self, x):
        out = self.pixel_attention(x)
        out = self.space_attention(out)
        return out


class LKA(nn.Module):
    def __init__(self, channels):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 7, padding='same', groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 13, stride=1, padding='same', groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class ESA(nn.Module):
    def __init__(self, channels):
        super(ESA, self).__init__()
        f = channels // 4
        self.conv1 = nn.Conv2d(channels, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class LKConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(channels)
        self.proj_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x
