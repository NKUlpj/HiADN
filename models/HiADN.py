# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: HiADN.py
@Author: nkul
@Date: 2023/4/10 下午1:56
"""

import torch
import torch.nn as nn
from models.Common import conv_block
from models.Component import HiAB
from models.Attention import ESA


class HiADN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1) -> None:
        super(HiADN, self).__init__()
        _hidden_channels = 32
        _block_num = 7
        self.fea_conv = nn.Conv2d(in_channels, _hidden_channels, 3, padding='same')

        # HiAB
        self.b1 = HiAB(channels=_hidden_channels)
        self.b2 = HiAB(channels=_hidden_channels)
        self.b3 = HiAB(channels=_hidden_channels)
        self.b4 = HiAB(channels=_hidden_channels)
        self.b5 = HiAB(channels=_hidden_channels)
        self.b6 = HiAB(channels=_hidden_channels)
        self.b7 = HiAB(channels=_hidden_channels)

        # reduce channels
        self.c = conv_block(_hidden_channels * _block_num, _hidden_channels, kernel_size=1, act_type='lrelu')
        self.LR_conv = nn.Conv2d(_hidden_channels, _hidden_channels, 3, padding='same')
        self.attn = ESA(channels=_hidden_channels)
        self.exit = nn.Conv2d(_hidden_channels, out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        out_fea = self.fea_conv(x)

        out_b1 = self.b1(out_fea)
        out_b2 = self.b2(out_b1)
        out_b3 = self.b3(out_b2)
        out_b4 = self.b4(out_b3)

        out_b5 = self.b5(out_b4)
        out_b6 = self.b6(out_b5)
        out_b7 = self.b6(out_b6)

        out_b = self.c(torch.cat([out_b1, out_b2, out_b3, out_b4, out_b5, out_b6, out_b7], dim=1))

        out_lr = self.LR_conv(out_b)  # + out_fea

        out_lr = self.attn(out_lr)
        output = self.exit(out_lr)
        return output
