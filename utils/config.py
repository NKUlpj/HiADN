# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: config.py
@Author: nkul
@Date: 2023/4/27 上午11:22 
"""
import logging


'''
log_level
1. logging.DEBUG
2. logging.INFO
3. logging.WARNING
4. logging.ERROR

'''
__log_level = logging.DEBUG
__log_format = '%(asctime)s [%(levelname)s] - %(message)s'

# the Root directory for all raw and processed data
root_dir = 'Datasets_NPZ'  # Example of root directory name

# 'train' and 'valid' can be changed for different train/valid set splitting
set_dict = {'train': [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21],
            'valid': [2, 6, 10, 12],
            'test': [4, 14, 16, 20],
            'K562_test': [3, 11, 19, 21],
            'CH12-LX_test': [4, 9, 15, 18],
            }  # 19

set_lr_cutoff = {
    16: 100,
    32: 70,
    100: 25
}


def set_log_config():
    logging.basicConfig(level=__log_level, format=__log_format)
