# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: NumpyToCooler.py
@Author: nkul
@Date: 2023/5/18 下午12:42 
@GitHub: https://github.com/nkulpj
"""
import numpy as np
import pandas as pd
from cooler.create import ArrayLoader
import cooler


def __numpy_to_cooler(_in_file, _out_file, _res, _chr):
    _heatmap = np.load(_in_file, allow_pickle=True)['hic']
    _shape = _heatmap.shape[0]
    _lst = [{"name": _chr, "length": _shape * _res}]
    _chrom_sizes = pd.DataFrame(_lst).set_index('name')['length']
    _bins = cooler.binnify(_chrom_sizes, _res)
    _iterator = ArrayLoader(_bins, _heatmap, chunksize=int(1e6))
    cooler.create_cooler(_out_file, _bins, _iterator, dtypes={"count": "float"}, assembly="hg19")


if __name__ == '__main__':
    in_file = '/home/nkul/Desktop/SR/16/r16_chr4.npz'
    out_file = './r16_chr4.cool'
    __numpy_to_cooler(in_file, out_file, _res=10000, _chr=4)
