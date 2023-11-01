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


def __data_prepare(_n, _npz_file, _chunk=32, _stride=32, _bound=301, _lr_cutoff=100, _hr_cutoff=255):
    _data = np.load(_npz_file, allow_pickle=True)
    _hic = _data['hic']
    full_size = _hic.shape[0]
    _hic = np.minimum(_hr_cutoff, _hic)
    _hic = _hic / np.max(_hic)
    div_hic, div_index = divide(_hic, _n, _chunk, _stride, _bound, verbose=True)
    return _n, div_hic, div_index, full_size


def __data_divider(_n, h_file, d_file, _chunk=32, _stride=32, _bound=301, _lr_cutoff=100, _hr_cutoff=255):
    high_data = np.load(h_file, allow_pickle=True)
    down_data = np.load(d_file, allow_pickle=True)

    high_hic = high_data['hic']
    down_hic = down_data['hic']
    compact_idx = high_data['compact']
    full_size = high_hic.shape[0]

    # Compacting
    high_hic = compact_matrix(high_hic, compact_idx)
    down_hic = compact_matrix(down_hic, compact_idx)

    # Clamping
    high_hic = np.minimum(_hr_cutoff, high_hic)
    down_hic = np.minimum(_lr_cutoff, down_hic)

    # Rescaling
    high_hic = high_hic / np.max(high_hic)
    down_hic = down_hic / _lr_cutoff

    # Split down sampled data
    div_d_hic, div_index = divide(down_hic, _n, _chunk, _stride, _bound)
    # @nkul there is no need for pooling
    # div_d_hic = pooling(div_d_hic, scale=1, pool_type='max', verbose=False).numpy()

    # Split high data
    div_h_hic, _ = divide(high_hic, _n, _chunk, _stride, _bound, verbose=True)
    return _n, div_d_hic, div_h_hic, div_index, compact_idx, full_size


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
    target = np.concatenate([r[2] for r in results])
    inds = np.concatenate([r[3] for r in results])
    compacts = {r[0]: r[4] for r in results}
    sizes = {r[0]: r[5] for r in results}

    filename = f'{cell_line}_c{chunk}_s{stride}_b{bound}_{postfix}.npz'
    split_file = os.path.join(out_dir, filename)
    np.savez_compressed(
        split_file,
        data=data,
        target=target,
        compacts=compacts,
        inds=inds,
        sizes=sizes)
    logging.debug(f'Saving file:{split_file}')