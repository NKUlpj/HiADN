# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: predict.py
@Author: nkul
@Date: 2023/4/10 下午2:00
"""
import sys

from utils.parser_helper import model_predict_parser
from utils.model_predict import model_predict


if __name__ == '__main__':
    args = model_predict_parser().parse_args(sys.argv[1:])
    model_name = args.model
    predict_file = args.predict_file
    batch_size = args.batch_size
    ckpt = args.ckpt
    model_predict(model_name, predict_file,  batch_size, ckpt)
