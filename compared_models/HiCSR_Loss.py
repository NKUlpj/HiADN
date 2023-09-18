# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: HiCSR_Loss.py
@Author: nkul
@Date: 2023/4/28 下午3:09
# Code was taken from https://github.com/PSI-Lab/HiCSR
"""
# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
from compared_models.HiCSR import DAE


class FeatureReconstructionLoss(nn.Module):
    def __init__(self):
        super(FeatureReconstructionLoss, self).__init__()
        loss_net = DAE()
        self.dae_path = os.path.join(self.__get_path(), 'pretrained_models/DAE.pth')
        loss_net.load_state_dict(torch.load(self.dae_path))
        encoder = list(loss_net.children())[0]

        self.sub_nets = []
        layers = [0, 1, 2, 3, 4]
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

    def forward(self, prediction, target):
        feature_loss = torch.tensor([0.0]).float().cuda()
        for net in self.sub_nets:
            pred_feat = net(prediction)
            target_feat = net(target)
            loss = self.loss(pred_feat, target_feat)
            feature_loss += loss
        return feature_loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.feature_loss = FeatureReconstructionLoss()
        self.loss_weight = [2.5e-3, 1, 1]

    def forward(self, out_images, target_images):
        image_loss = self.l1_loss(out_images, target_images)
        feature_loss = sum(self.feature_loss(out_images, target_images))
        return image_loss + feature_loss
