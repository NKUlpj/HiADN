# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: Common.py
@Author: nkul
@Date: 2023/4/10 下午1:50
"""
from collections import OrderedDict
from models.Attention import *


# ================== private func start ==========================
def __get_norm(norm_type, hidden_channels):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(hidden_channels, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(hidden_channels, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def __get_pad_num(kernel_size, dilation):
    r"""
    return padding size
    Assume no dilation i.e. dilation = 1
    then
    (n + 2p - k) / s + 1 = n
    s = 1
    --> p = (k - 1) // 2
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def __get_pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def __get_sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_act_fn(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
# ================== private func end ==========================


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    r"""
    func   -  auto calculate padding,
    return -  Conv2d, in size == out size
    ---
    (n + 2p - k )/ s + 1 = n
    2p - k =  - 1
    p = (k - 1) // 2
    """
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups
    )


def conv_block(in_channels, hidden_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = __get_pad_num(kernel_size, dilation)
    p = __get_pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = get_act_fn(act_type) if act_type else None
    n = __get_norm(norm_type, hidden_channels) if norm_type else None
    return __get_sequential(p, c, n, a)
