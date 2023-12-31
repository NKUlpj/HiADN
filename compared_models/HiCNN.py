# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: HiCNN.py
@Author: nkul
@Date: 2023/4/10 下午1:34
# Code was taken from http://dna.cs.miami.edu/HiCNN2/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLUBlock(nn.Module):
    def __init__(self):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net1_conv1 = nn.Conv2d(1, 64, 13)
        self.net1_conv2 = nn.Conv2d(64, 64, 1)
        self.net1_conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.net1_conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.net1_conv5 = nn.Conv2d(128 * 25, 1000, 1, padding=0, bias=True)
        self.net1_conv6 = nn.Conv2d(1000, 64, 1, padding=0, bias=True)
        self.net1_conv7 = nn.Conv2d(64, 1, 3, padding=1, bias=False)

        self.net2_conv1 = nn.Conv2d(1, 8, 13)
        self.net2_conv2 = nn.Conv2d(8, 1, 1)
        self.residual_layer_vdsr = self.make_layer(ConvReLUBlock, 18)
        self.input_vdsr = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_vdsr = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.net3_conv1 = nn.Conv2d(1, 8, 9)
        self.net3_conv2 = nn.Conv2d(8, 8, 1)
        self.net3_conv3 = nn.Conv2d(8, 1, 5)

        self.relu = nn.ReLU(inplace=True)

        self.weights = nn.Parameter((torch.ones(1, 3) / 3), requires_grad=True)
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        # ConvNet1
        x = self.relu(self.net1_conv1(input))
        x = self.relu(self.net1_conv2(x))
        residual = x
        x2 = self.net1_conv3(x)
        output1 = x2
        outtmp = []
        for i in range(25):
            output1 = self.net1_conv4R(self.relu(self.net1_conv4R(self.relu(output1))))
            output1 = torch.add(output1, x2)
            outtmp.append(output1)
        output1 = torch.cat(outtmp, 1)
        output1 = self.net1_conv5(output1)
        output1 = self.net1_conv6(output1)
        output1 = torch.add(output1, residual)
        output1 = self.net1_conv7(output1)

        # ConvNet2
        x_vdsr = self.relu(self.net2_conv1(input))
        x_vdsr = self.relu(self.net2_conv2(x_vdsr))
        residual2 = x_vdsr
        output2 = self.relu(self.input_vdsr(x_vdsr))
        output2 = self.residual_layer_vdsr(output2)
        output2 = self.output_vdsr(output2)
        output2 = torch.add(output2, residual2)

        # ConvNet3
        output3 = self.net3_conv1(input)
        output3 = F.relu(output3)
        output3 = self.net3_conv2(output3)
        output3 = F.relu(output3)
        output3 = self.net3_conv3(output3)
        output3 = F.relu(output3)

        # w1*output1 + w2*output2 + w3*output3
        w_sum = self.weights.sum(1)
        output = (output1 * self.weights.data[0][0] / w_sum) + (
            output2 * self.weights.data[0][1] / w_sum) + (output3 * self.weights.data[0][2] / w_sum)

        return output
