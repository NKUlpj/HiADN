# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: visual.py
@Author: nkul
@Date: 2023/5/8 下午2:20 
@GitHub: https://github.com/nkulpj
"""
import logging
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.parser_helper import model_visual_parser
from utils.config import root_dir


__img_dir = os.path.join(root_dir, 'img')


def __plot_hic(matrix, start, end, percentile, name, cmap, color_bar=False):
    data2d = matrix[start:end, start:end]
    v_max = np.percentile(data2d, percentile)
    fig, ax = plt.subplots()
    im = ax.imshow(data2d, interpolation="nearest", vmax=v_max, vmin=0, cmap=cmap)
    if name is None or name == "":
        name = f"{start} - {end}"
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if color_bar:
        fig.colorbar(im, ax=ax)

    __save_file = os.path.join(__img_dir, f'{name}.png')
    plt.savefig(__save_file)
    logging.info(f'Save pic to {__save_file}')
    plt.show()


if __name__ == '__main__':
    logging.disable(logging.DEBUG)
    args = model_visual_parser().parse_args(sys.argv[1:])
    _file = args.file
    _s = int(args.start)
    _e = int(args.end)
    _p = int(args.percentile)
    _n = args.name
    _m = np.load(_file)['hic']
    _c = args.cmap
    __plot_hic(_m, _s, _e, _p, _n, _c, color_bar=True)
