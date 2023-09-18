# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: train.py
@Author: nkul
@Date: 2023/4/10 下午2:00
"""
import sys
from utils.parser_helper import model_train_parser
from utils.model_train import model_train


if __name__ == '__main__':
    args = model_train_parser().parse_args(sys.argv[1:])
    model_name = args.model
    train_file = args.train_file
    valid_file = args.valid_file
    max_epochs = args.epochs
    batch_size = args.batch_size
    verbose = (args.verbose == 1)
    model_train(model_name, train_file, valid_file, max_epochs, batch_size, verbose)
