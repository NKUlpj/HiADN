# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: read_data.py
@Author: nkul
@Date: 2023/5/13 上午11:11 
@GitHub: https://github.com/nkulpj

A tool to read raw data from Rao's Hi-C experiment
In root dir(eg. Datasets_NPZ), your file must be like below
mat is created by running this script
Datasets_NPZ
├── raw
│   ├── K562
│   ├── GM12878
│   └── CH12-LX
├── mat
│   ├── K562
│   │   ├── chr1_10kb.npz
│   │   ├── chr1_40kb.npz
│   │   └── ...
│   ├── GM12878
│   └── CH12-LX

output: .npz
"""

import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import data_read_parser, res_map, mkdir
from utils.io_helper import *
import numpy as np
import time

import logging
from utils.config import set_log_config, root_dir
set_log_config()


def __read_data(data_file, norm_file, _out_dir, _resolution):
    filename = os.path.basename(data_file).split('.')[0] + '.npz'
    out_file = os.path.join(_out_dir, filename)
    try:
        _hic, _idx = read_coo2mat(data_file, norm_file, _resolution)
    except NotImplementedError:
        logging.error(f'Abnormal file: {norm_file}')
        exit()
    np.savez_compressed(out_file, hic=_hic, compact=_idx)
    logging.debug(f'Saving file:{out_file}')


if __name__ == '__main__':
    args = data_read_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    resolution = args.high_res
    map_quality = args.map_quality
    postfix = [args.norm_file, 'RAWobserved']
    raw_dir = os.path.join(root_dir, 'raw', cell_line)

    norm_files = []
    data_files = []
    for root, dirs, files in os.walk(raw_dir):
        if len(files) > 0:
            if (resolution in root) and (map_quality in root):
                for f in files:
                    if f.endswith(postfix[0]):
                        norm_files.append(os.path.join(root, f))
                    elif f.endswith(postfix[1]):
                        data_files.append(os.path.join(root, f))

    out_dir = os.path.join(root_dir, 'mat', cell_line)
    mkdir(out_dir)
    logging.debug(f'Start reading data, there are {len(norm_files)} files ({resolution}).')
    logging.debug(f'Output directory: {out_dir}')

    start = time.time()
    for data_fn, norm_fn in zip(data_files, norm_files):
        __read_data(data_fn, norm_fn, out_dir, res_map[resolution])
    logging.debug(f'All reading processes done. Running cost is {(time.time()-start)/60:.1f} min.')
