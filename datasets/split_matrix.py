# -*- coding: UTF-8 -*-
"""
@Project: HiADN 
@File: split_matrix.py
@Author: nkul
@Date: 2023/11/1 下午2:45 
@GitHub: https://github.com/nkulpj
"""
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import matrix_prepare_parser, mkdir
from utils.io_helper import divide, compact_matrix
import numpy as np
import time
import logging
from utils.config import set_log_config, root_dir, set_dict, set_lr_cutoff
set_log_config()


def __data_prepare(_n, _npz_file, _chunk=40, _stride=40, _bound=201, _lr_cutoff=100, _hr_cutoff=255):
    _data = np.load(_npz_file, allow_pickle=True)
    _hic = _data['hic']
    full_size = _hic.shape[0]
    # _cutoff = np.percentile(_hic, 99)
    # _hic = np.minimum(_cutoff, _hic)
    _hic = _hic / np.max(_hic)
    div_hic, div_index = divide(_hic, _n, _chunk, _stride, _bound, verbose=True)
    return _n, div_hic, div_index, full_size


if __name__ == '__main__':
    # 将npz转化为tensor 输入到模型中
    args = matrix_prepare_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res

    dataset = args.dataset
    chunk = args.chunk
    stride = args.stride
    bound = args.bound

    chr_list = set_dict[dataset]
    postfix = cell_line.lower() if dataset == 'all' else dataset

    logging.debug(f'Going to read matrix')

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    out_dir = os.path.join(root_dir, 'data')
    mkdir(out_dir)

    start = time.time()
    results = []
    for n in chr_list:
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        # res = __data_divider(n, high_file, down_file, chunk, stride, bound, _lr_cutoff=lr_cutoff)
        res = __data_prepare(n, high_file, chunk, stride, bound)
        results.append(res)

    logging.debug(f'All data generated. Running cost is {(time.time()-start)/60:.1f} min.')
    data = np.concatenate([r[1] for r in results])
    inds = np.concatenate([r[2] for r in results])
    sizes = {r[0]: r[3] for r in results}

    filename = f'{cell_line}_c{chunk}_s{stride}_b{bound}_{postfix}.npz'
    split_file = os.path.join(out_dir, filename)
    np.savez_compressed(
        split_file,
        data=data,
        inds=inds,
        sizes=sizes)
    logging.debug(f'Saving file:{split_file}')
