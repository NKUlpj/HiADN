# -*- coding: UTF-8 -*-
"""
@Project: HiADN 
@File: numpyToTxT.py
@Author: nkul
@Date: 2023/10/27 上午11:25 
@GitHub: https://github.com/nkulpj
"""
import cooler
import numpy as np


def numpy2txt(np_path, txt_path):
    m = np.load(np_path, allow_pickle=True)['hic']
    shape = m.shape[0]
    with open(txt_path, 'w') as f:
        for i in range(shape):
            for j in range(i, shape):
                if m[i][j] == 0.0:
                    continue
                f.write(f'{i}\t{j}\t{m[i][j] * 100}\n')


if __name__ == '__main__':
    # np_path = '/home/nkul/Desktop/ex2/HR/npz/GM12878_r16_chr4.npz'
    # txt_path = '10K/4_4.txt'
    # numpy2txt(np_path, txt_path)

    c = cooler.Cooler('/home/nkul/Desktop/ex2/HR/cool/r16_chr4.cool')
    cooler.balance_cooler(c)

