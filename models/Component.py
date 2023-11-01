# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: Component.py
@Author: nkul
@Date: 2023/4/24 上午11:14 
ALL IN ONE
"""
import torch
import torch.nn as nn

from models.Common import *


class HiFM(nn.Module):
    def __init__(self, channels, k=4) -> None:
        super(HiFM, self).__init__()
        r'''
        in:         [C * W * H]
        out:        [C/2 * W * H]
        '''
        self.k = k
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.k, stride=self.k),
            nn.Conv2d(channels, channels, 3, padding='same'),
            nn.GELU(),
            nn.Upsample(scale_factor=self.k, mode='nearest')
        )
        self.attn = HiCBAM(channels * 2)
        self.out = nn.Conv2d(channels * 2, channels // 2, 1, padding='same')

    def forward(self, x):
        tl = self.net(x)
        tl = x - tl
        x = torch.cat((x, tl), 1)
        x = self.attn(x)
        x = self.out(x)
        return x


class HiAB(nn.Module):
    def __init__(self, channels):
        super(HiAB, self).__init__()
        _hidden_channels = channels // 2

        self.hifm1 = HiFM(channels)
        self.c1_r = conv_layer(channels, channels, 3)

        self.hifm2 = HiFM(channels)
        self.c2_r = conv_layer(channels, channels, 3)

        self.hifm3 = HiFM(channels)
        self.c3_r = conv_layer(channels, channels, 3)

        # self.c4 = conv_layer(channels, _hidden_channels, 3)
        self.hifm4 = HiFM(channels)

        self.act = nn.GELU()
        self.c5 = conv_layer(channels * 2, channels, 1)
        self.attn = LKConv(channels)
        # self.attn = LKA(channels=channels)

    def forward(self, x):
        distilled_c1 = self.act(self.hifm1(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(r_c1 + x)

        distilled_c2 = self.act(self.hifm2(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.hifm3(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)

        # r_c4 = self.act(self.c4(r_c3))
        distilled_c4 = self.act(self.hifm4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        out_fused = self.attn(self.c5(out))
        # there is no global residual connection
        return out_fused + x
