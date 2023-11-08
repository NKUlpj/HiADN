# -*- coding: UTF-8 -*-
"""
@Project: HiADN 
@File: HiADN_Loss.py
@Author: nkul
@Date: 2023/11/3 下午4:00 
@GitHub: https://github.com/nkulpj
"""
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


# Code from HiCSR
class DAE(nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(DAE, self).__init__()
        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []

        conv_layers.append(
            nn.Sequential(
                nn.Conv2d(1, num_features, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        for i in range(num_layers - 1):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        for i in range(num_layers - 1):
            deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        deconv_layers.append(
            nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = torch.tanh(x)
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * 1 * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class VGG_Loss(nn.Module):
    def __init__(self, weight=1.0, layer=35):
        super().__init__()
        pretrained_vgg = vgg16(pretrained=True)
        modules = [m for m in pretrained_vgg.features]
        self.vgg = nn.Sequential(*modules[:layer])
        self.weight = weight

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)

        self.sub_mean = MeanShift(vgg_mean, vgg_std)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_hr, vgg_sr)
        return self.weight * loss


class FeatureReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_nets = []
        loss_net = DAE()
        self.dae_path = os.path.join(self.__get_path(), 'pretrained_models/DAE.pth')
        loss_net.load_state_dict(
            torch.load(self.dae_path)
        )

        encoder = list(loss_net.children())[0]
        layers = [_ for _ in range(5)]
        self.layers = layers
        for layer in layers:
            list_of_layers = list(encoder)[:layer]
            final_layer = [encoder[layer][0]]
            sub_net = nn.Sequential(*(list_of_layers + final_layer)).float().eval().cuda()
            for param in sub_net.parameters():
                param.requires_grad = False
            self.sub_nets.append(sub_net)
        self.loss = nn.MSELoss()

    @staticmethod
    def __get_path():
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        return cur_dir

    def forward(self, out_images, target_images):
        feature_loss = torch.tensor([0.0]).float().cuda()
        for net in self.sub_nets:
            pred_feat = net(out_images)
            target_feat = net(target_images)
            loss = self.loss(pred_feat, target_feat)
            feature_loss += loss
        return feature_loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGG_Loss(weight=0.0005, layer=8)
        self.feature_loss = FeatureReconstructionLoss()

    def forward(self, out_images, target_images):
        image_loss = self.l1_loss(out_images, target_images)
        feature_loss = sum(self.feature_loss(out_images, target_images))
        vgg_sr = out_images.repeat([1, 3, 1, 1])
        vgg_hr = target_images.repeat([1, 3, 1, 1])
        perception_loss = self.vgg_loss(vgg_sr, vgg_hr)
        return image_loss + feature_loss + perception_loss
