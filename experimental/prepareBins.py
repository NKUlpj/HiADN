# -*- coding: UTF-8 -*-
"""
@Project: HiADN 
@File: prepareBins.py
@Author: nkul
@Date: 2023/10/18 上午11:31 
@GitHub: https://github.com/nkulpj
"""

'''
prepare the Bin.bed fro genome disco
GM12878: 4 14 16 20
'''
# 先求染色体大小
import numpy as np
bin_size = 10 * 1000  # 10kb
chr_4 = 19120


def write_bed_file(chr_num, chr_size, idx):
    with open(f'Bins.w10000.{idx}.bed', 'w') as f:
        for i in range(chr_size * idx , chr_size * (idx + 1)):
            line = f"chr{chr_num}\t{i * bin_size}\t{(i + 1) * bin_size}\t{i * bin_size}\n"
            f.write(line)


def write_hic_file(matrix, chr_num, idx):
    # shape = matrix.shape[0]
    shape = 200
    s = shape * idx
    e = shape * (idx + 1)
    with open(f"HIC002.{idx}.res10000", 'w') as f:
        for i in range(s, e):
            for j in range(i, e):
                score = round(matrix[i][j] * 10, 0)
                if score < 0:
                    score = 0
                if score == 0.0:
                    continue
                line = f"{chr_num}\t{i * bin_size}\t{chr_num}\t{j * bin_size}\t{score}\n"
                f.write(line)


if __name__ == "__main__":
    lr_path = '/home/nkul/Desktop/HiADN/Datasets_NPZ/predict/HiADN/GM12878_c40_s40_b201_r32_test_chr4.npz'
    # lr_path = '/home/nkul/Desktop/ex2/HR/npz/GM12878_r16_chr4.npz'
    lr = np.load(lr_path, allow_pickle=True)['hic']
    # write_bed_file(4, 200)
    for i in range(90):
        write_hic_file(lr, 4, i)
        # write_bed_file(4, 200, i)

