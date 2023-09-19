# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: evaluate.py.py
@Author: nkul
@Date: 2023/9/19 下午1:02 
@GitHub: https://github.com/nkulpj
"""

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from .util_func import get_device

device = get_device()

betas = (0.0448, 0.2856)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, betas=betas).to(device)


def evaluate_ms_ssim(x, y):
    ms_ssim_val = ms_ssim(x, y)
    return ms_ssim_val
