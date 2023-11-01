# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: util_func.py
@Author: nkul
@Date: 2023/4/10 下午2:07
"""


import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import compared_models.HiCARN_1 as HiCARN
import compared_models.HiCNN as HiCNN
import compared_models.DeepHiC as DeepHiC
import compared_models.HiCSR as HiCSR
import compared_models.HiCARN_1_Loss as HiCARN_1_Loss
import compared_models.DeepHiC_Loss as DeepHiC_Loss
import compared_models.HiCSR_Loss as HiCSR_Loss

import models.HiADN as HiADN
import models.HiADN_Loss as HiADN_Loss

import logging
from utils.config import set_log_config, root_dir
set_log_config()


# get model by name
def get_model(_model_name):
    _padding = False
    _netG = None
    _netD = None

    if _model_name == 'HiADN':
        _netG = HiADN.HiADN()

    elif _model_name == 'HiCARN':
        _netG = HiCARN.Generator(num_channels=64)

    elif _model_name == 'HiCNN':
        _netG = HiCNN.Generator()
        _padding = True

    elif _model_name == 'HiCSR':
        _padding = True
        _netG = HiCSR.Generator()
        _netD = HiCSR.Discriminator()

    elif _model_name == 'DeepHiC':
        _padding = False
        _netG = DeepHiC.Generator()
        _netD = DeepHiC.Discriminator(in_channel=1)
    else:
        raise NotImplementedError('Model {} is not implemented'.format(_model_name))
    logging.debug(f'Running {_model_name}')
    return _netG, _padding, _netD


# get data loader
def loader(file_name, loader_type='Train', padding=False, shuffle=True, batch_size=64):
    __data_dir = os.path.join(root_dir, 'data')
    __file = os.path.join(__data_dir, file_name)
    __file_np = np.load(__file)

    __input_np = __file_np['data']
    __input_tensor = torch.tensor(__input_np, dtype=torch.float)
    if padding:
        __input_tensor = F.pad(__input_tensor, (6, 6, 6, 6), mode='constant')

    __inds_np = __file_np['inds']
    __inds_tensor = torch.tensor(__inds_np, dtype=torch.long)
    logging.debug(f"{loader_type} Set Size - {__input_tensor.size()}")

    __has_target = False

    if 'target' in __file_np.keys():
        __has_target = True
        __target_np = __file_np['target']
        __target_tensor = torch.tensor(__target_np, dtype=torch.float)
        __dataset = TensorDataset(__input_tensor, __target_tensor, __inds_tensor)
        __data_loader = DataLoader(__dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        __dataset = TensorDataset(__input_tensor, __inds_tensor)
        __data_loader = DataLoader(__dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return __data_loader, __has_target


def get_device():
    _has_cuda = torch.cuda.is_available()
    _device = torch.device('cuda:0' if _has_cuda else 'cpu')
    logging.debug(f"CUDA available? {_has_cuda}")
    if not _has_cuda:
        logging.warning("GPU acceleration is strongly recommended")
    return _device


def get_d_loss_fn(_model_name):
    if _model_name == 'DeepHiC':
        logging.debug(f"Using BCELoss as D_Loss")
        return torch.nn.BCELoss()
    elif _model_name == 'HiCSR':
        logging.debug(f"Using BCEWithLogitsLoss as D_Loss")
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f'{_model_name} has not implemented')


def get_loss_fn(_model_name, device='cpu'):
    if _model_name == 'HiADN':
        logging.debug('Using HiADN_Loss')
        loss = HiADN_Loss.LossL(device=device)
    elif _model_name == 'DeepHiC':
        logging.debug('Using DeepHiC_Loss')
        loss = DeepHiC_Loss.GeneratorLoss()
    elif _model_name == 'HiCSR':
        logging.debug('Using HiCSR_Loss')
        loss = HiCSR_Loss.GeneratorLoss()
    elif _model_name == 'HiCNN':
        logging.debug('Using HiCNN_Loss')
        loss = nn.MSELoss()
    else:
        logging.debug('Using HiCARN_Loss')
        loss = HiCARN_1_Loss.GeneratorLoss()
    return loss
