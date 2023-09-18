# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: HiRDN_Loss.py
@Author: nkul
@Date: 2023/4/24 下午12:00 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
import warnings

warnings.filterwarnings("ignore")


class LossL(nn.Module):
    """
    Loss_L = [r1 * vgg(3) + r2 * vgg(8) + r3 * vgg(15)] + alpha * dists_loss + beta * MS_SSIM_L1_LOSS
    MS_SSIM_L1_LOSS = beta1 * loss_ms_ssim + (1 - beta1) * gaussian_l1
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ms_ssim_l1_loss = MS_SSIM_L1_LOSS(device=device, alpha=0.84, weight=1)
        self.vgg_loss = VGG_Loss(weight=0.01, layer=8)

    def forward(self, out_images, target_images):
        vgg_sr = out_images.repeat([1, 3, 1, 1])
        vgg_hr = target_images.repeat([1, 3, 1, 1])
        perception_loss = self.vgg_loss(vgg_sr, vgg_hr)
        ms_ssim_l1_loss = self.ms_ssim_l1_loss(out_images, target_images)
        return ms_ssim_l1_loss + perception_loss


class MS_SSIM_L1_LOSS(nn.Module):
    """
    Some Code from https://github.com/psyrocloud/MS-SSIM_L1_LOSS
    Paper "Loss Functions for Image Restoration With Neural Networks"
    """

    def __init__(self, device, data_range=1.0, alpha=0.84, weight=1., channel=1):
        super(MS_SSIM_L1_LOSS, self).__init__()
        k = (0.01, 0.03)
        gaussian_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
        self.channel = channel
        self.DR = data_range
        self.C1 = (k[0] * data_range) ** 2
        self.C2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.weight = weight
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 1, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.to(device=device)

    @staticmethod
    def _f_special_gauss_1d(size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _f_special_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma ([float]): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._f_special_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel
        mu_x = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)
        mu_y = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mu_x2
        sigma_y2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - mu_y2
        sigma_xy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - mu_xy

        # l(j), cs(j) in MS-SSIM
        _l = (2 * mu_xy + self.C1) / (mu_x2 + mu_y2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigma_xy + self.C2) / (sigma_x2 + sigma_y2 + self.C2)

        if self.channel == 3:
            l_m = _l[:, -1, :, :] * _l[:, -2, :, :] * _l[:, -3, :, :]
        else:
            l_m = _l[:, -1, :, :]

        # l_m = _l[:, -1, :, :] * _l[:, -2, :, :] * _l[:, -3, :, :]
        p_ics = cs.prod(dim=1)

        loss_ms_ssim = 1 - l_m * p_ics  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        return self.weight * loss_mix.mean()


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
