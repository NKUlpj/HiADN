# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: io_helper.py
@Author: nkul
@Date: 2023/4/10 下午12:42
# Some Code was taken from https://github.com/omegahh/DeepHiC
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import logging
from utils.config import set_log_config
set_log_config()


except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}


def read_coo2mat(coo_file, norm_file, resolution):
    """
    Function used for reading a coordinated tag file to a square matrix.
    """
    norm = open(norm_file, 'r').readlines()
    norm = np.array(list(map(float, norm)))
    compact_idx = list(np.where(np.isnan(norm) ^ True)[0])
    pd_mat = pd.read_csv(coo_file, sep='\t', header=None, dtype=np.int32)
    row = pd_mat[0].values // resolution
    col = pd_mat[1].values // resolution
    val = pd_mat[2].values
    # here is a full HiC Matrix
    mat = coo_matrix((val, (row, col)), shape=(len(norm), len(norm))).toarray()
    mat = mat.astype(float)
    norm[np.isnan(norm)] = 1
    mat = mat / norm
    mat = mat.T / norm
    _hic = mat + np.tril(mat, -1).T
    return _hic.astype(np.int32), compact_idx


def __dense2tag(matrix):
    """
    Converts a square matrix (dense) to coo-based tag matrix.
    """
    matrix = np.triu(matrix)
    tag_len = int(np.sum(matrix))
    tag_mat = np.zeros((tag_len, 2), dtype=np.int64)
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + int(data[i])
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len


def __tag2dense(tag, n_size):
    """
    Coverts a coo-based tag matrix to dense square matrix.
    """
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(n_size, n_size)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat


def down_sampling(matrix, down_ratio, verbose=False):
    """
    Down sampling method.
    """
    if verbose:
        logging.debug(f"[Down sampling] Matrix shape is {matrix.shape}")
    tag_mat, tag_len = __dense2tag(matrix)
    sample_idx = np.random.choice(tag_len, tag_len // down_ratio)
    sample_tag = tag_mat[sample_idx]
    if verbose:
        logging.debug(f'[Down sampling] Sampling 1/{down_ratio} of {tag_len} reads')
    down_mat = __tag2dense(sample_tag, matrix.shape[0])
    return down_mat


def divide(mat, chr_num, chunk_size=64, stride=64, bound=201, padding=True, species='hsa', verbose=False):
    """
    Dividing method.
    """
    chr_str = str(chr_num)
    if isinstance(chr_num, str):
        chr_num = except_chr[species][chr_num]
    result = []
    index = []
    size = mat.shape[0]
    if stride < chunk_size and padding:
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i - j) <= bound and i + chunk_size < height and j + chunk_size < width:
                sub_image = mat[i:i + chunk_size, j:j + chunk_size]
                result.append([sub_image])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose:
        logging.debug(f'[Chr{chr_str}] Dividing HiC matrix ({size}x{size}) into {len(result)} samples '
                      f'with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


def compact_matrix(matrix, compact_idx, verbose=False):
    """
    Compacts the matrix according to the index list.
    """
    compact_size = len(compact_idx)
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    if verbose:
        logging.debug(f'Compacting a {matrix.shape} shaped matrix to{result.shape} shaped!')
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    return result


def spread_matrix(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """
    Spreads the matrix according to the index list (a reversed operation to compactM).
    """
    result = np.zeros((full_size, full_size)).astype(c_mat.dtype)
    if convert_int:
        result = result.astype(np.int32)
    if verbose:
        logging.debug(f'Spreading a{c_mat.shape} shaped matrix to{result.shape} shaped!')
    for i, s_idx in enumerate(compact_idx):
        result[s_idx, compact_idx] = c_mat[i]
    return result


def together(mat_list, indices, corp=0, species='hsa', tag='HiC'):
    """
    Constructs a full dense matrix.
    """
    chr_nums = sorted(list(np.unique(indices[:, 0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]:
        chr_nums[-1] = except_chr[species][chr_nums[-1]]
    logging.debug(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = mat_list[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:, 0] == num)[0]
        sub_mats = mat_list[loci]
        index = indices[loci]
        width = index[0, 1]
        width = int(width)
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = int(pos[-2]), int(pos[-1])
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[i:i + h, j:j + w] = sub
        results[n] = full_mat
    return results
