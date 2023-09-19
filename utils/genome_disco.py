# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: genome_disco.py
@Author: nkul
@Date: 2023/4/10 下午12:41
# Code taken from https://github.com/kundajelab/genomedisco
"""


import gzip
import copy
from sklearn import metrics
import scipy.sparse as sps


def to_transition(together):
    sums = together.sum(axis=1)
    # make the ones that are 0, so that we don't divide by 0
    sums[sums == 0.0] = 1.0
    d = sps.spdiags(1.0 / sums.flatten(),
                    [0], together.shape[0], together.shape[1], format='csr')
    return d.dot(together)


def random_walk(m_input, t):
    # return m_input.__pow__(t)
    # return np.linalg.matrix_power(m_input,t)
    return m_input.__pow__(t)


def write_diff_vector_bed_file(diff_vector, nodes, nodes_idx, out_filename):
    out = gzip.open(out_filename, 'w')
    for i in range(diff_vector.shape[0]):
        node_name = nodes_idx[i]
        node_dict = nodes[node_name]
        out.write(str(node_dict['chr']) +
                  '\t' +
                  str(node_dict['start']) +
                  '\t' +
                  str(node_dict['end']) +
                  '\t' +
                  node_name +
                  '\t' +
                  str(diff_vector[i][0]) +
                  '\n')
    out.close()


def compute_reproducibility(m1_csr, m2_csr, transition, t_max=3, t_min=3):
    # make symmetric
    m1up = m1_csr
    m1down = m1up.transpose()
    m1 = m1up + m1down

    m2up = m2_csr
    m2down = m2up.transpose()
    m2 = m2up + m2down

    # convert to an actual transition matrix
    if transition:
        m1 = to_transition(m1)
        m2 = to_transition(m2)

    # count nonzero nodes (note that we take the average number of nonzero
    # nodes in the 2 datasets)
    row_sums_1 = m1.sum(axis=1)
    nonzero_1 = [i for i in range(row_sums_1.shape[0]) if row_sums_1[i] > 0.0]
    row_sums_2 = m2.sum(axis=1)
    nonzero_2 = [i for i in range(row_sums_2.shape[0]) if row_sums_2[i] > 0.0]
    # nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
    nonzero_total = 0.5 * (1.0 * len(list(set(nonzero_1))) +
                           1.0 * len(list(set(nonzero_2))))

    scores = []
    if True:
        for t in range(1, t_max + 1):  # range(args.t_min,args.t_max+1):
            rw1 = m1
            rw2 = m2
            if t == 1:
                rw1 = copy.deepcopy(m1)
                rw2 = copy.deepcopy(m2)
            else:
                rw1 = rw1.dot(m1)
                rw2 = rw2.dot(m2)

            if t >= t_min:
                diff = abs(rw1 - rw2).sum()
                scores.append(1.0 * float(diff) / float(nonzero_total))

    # compute final score
    ts = range(t_min, t_max + 1)
    if t_min == t_max:
        auc = scores[0]
        if 2 < auc:
            auc = 2
        elif 0 <= auc <= 2:
            auc = auc
    else:
        auc = metrics.auc(range(len(ts)), scores) / (len(ts) - 1)
    reproducibility = 1 - auc
    return reproducibility
