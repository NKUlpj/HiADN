# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: dataConvert.py
@Author: nkul
@Date: 2023/5/15 下午4:12 
@GitHub: https://github.com/nkulpj
"""
import numpy as np


def numpy_to_top_dom(_matrix_file, _out_file, _chr_num: int):
    m = np.load(_matrix_file, allow_pickle=True)['hic']
    f = open(_out_file, "w")
    shape = m.shape[0]
    for i in range(shape):
        line = 'chr' + str(_chr_num) + '\t' + str(shape * i) + '\t' + str(shape * i + shape)
        for j in range(shape):
            line += '\t' + str(m[i][j])
        line += '\r\n'
        f.write(line)
    f.close()


def process_top_dom_out(_in_file, _out_file):
    nf = open(_out_file, "w")
    with open(_in_file, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            t = line.split('\t')
            if t[5] == 'domain':
                if int(t[3]) - int(t[1]) < 5:
                    continue
                else:
                    nf.write(t[1] + '\t' + t[3] + '\n')
    nf.close()


if __name__ == '__main__':
    # matrix_file = '/home/nkul/Desktop/SR/16/Predict_HiCARN_GM12878_c64_s64_b201_r16_test_chr4.npz'
    # out_file = '/home/nkul/Desktop/SR/16/HiCARN_chr4_top_dom_in'
    # chr_num = 4
    # numpy_to_top_dom(matrix_file, out_file, chr_num)

    in_file = '/home/nkul/Desktop/SR/16/LR_chr4_top_out.domain'
    out_file = './LR_top_out'
    process_top_dom_out(in_file, out_file)
