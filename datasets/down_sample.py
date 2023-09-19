# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: down_sample.py
@Author: nkul
@Date: 2023/5/13 上午11:18 
@GitHub: https://github.com/nkulpj
"""
import numpy as np
import time
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import *
from utils.io_helper import down_sampling
import logging
from utils.config import set_log_config, root_dir
set_log_config()


def __down_sample(in_file, _low_res, _ratio):
    data = np.load(in_file, allow_pickle=True)
    hic = data['hic']
    compact_idx = data['compact']
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(in_file), f'{chr_name}_{_low_res}.npz')
    down_hic = down_sampling(hic, _ratio)

    np.savez_compressed(out_file, hic=down_hic, compact=compact_idx, ratio=_ratio)
    logging.debug(f'Saving file:{out_file}')


if __name__ == '__main__':
    args = data_down_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    high_res = args.high_res
    ratio = args.ratio
    low_res = f"{ratio}ds"
    data_dir = os.path.join(root_dir, 'mat', cell_line)
    in_files = [os.path.join(data_dir, f)
                for f in os.listdir(data_dir) if f.find(high_res + '.npz') >= 0]
    logging.debug(f'find files:{in_files}')
    logging.debug(f'Generating {low_res} files from {high_res} files by {ratio}x down_sampling.')
    start = time.time()
    for file in in_files:
        __down_sample(file, f"{high_res}_{low_res}", ratio)
    logging.debug(f'All down_sampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')
