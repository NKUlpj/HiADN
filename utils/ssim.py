# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: ssim.py
@Author: nkul
@Date: 2023/4/10 下午12:50
# Code was taken from https://github.com/OluwadareLab/HiCARN/
"""


from math import exp

import torch
import torch.nn.functional as F


def _to_img(mat):
    m = torch.tensor(mat)
    # convert to float and add channel dimension
    return m.float()


def _to_hic(mat):
    mat.squeeze_()
    return mat.numpy()


def gaussian(width, sigma):
    gauss = torch.Tensor(
        [exp(-(x - width // 2)**2 / float(2 * sigma**2)) for x in range(width)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel,
        1,
        window_size,
        window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1,
        window,
        padding=window_size // 2,
        groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2,
        window,
        padding=window_size // 2,
        groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2,
        window,
        padding=window_size // 2,
        groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size()[1]
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
