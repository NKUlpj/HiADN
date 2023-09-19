# -*- coding: UTF-8 -*-
"""
@Project: HiADN
@File: parser_helper.py
@Author: nkul
@Date: 2023/4/10 下午12:30
"""
import os
import argparse
import textwrap
import logging
from .config import set_log_config
set_log_config()

res_map = {
    '5kb': 5_000,
    '10kb': 10_000,
    '25kb': 25_000,
    '50kb': 50_000,
    '100kb': 100_000,
    '250kb': 250_000,
    '500kb': 500_000,
    '1mb': 1_000_000}


help_opt = (('--help', '-h'), {
    'action': 'help',
    'help': "Print this help message and exit"})


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        logging.debug(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


# chr12_10kb.npz -> predict_chr13_40kb.npz
def chr_num_str(x):
    start = x.find('chr')
    part = x[start + 3:]
    end = part.find('_')
    return part[:end]


# X -> 23 | str(12) -> int(12)
def chr_digit(filename):
    chr_n = chr_num_str(os.path.basename(filename))
    if chr_n == 'X':
        n = 23
    else:
        n = int(chr_n)
    return n


def data_read_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            A tools to read raw data from Rao's Hi-C experiment.
            ------------------------------------------------------
            Use example : python ./datasets/read_prepare.py -c GM12878
            ------------------------------------------------------
        '''
                                    ),
        add_help=False)

    req_args = parser.add_argument_group('Required Arguments')

    req_args.add_argument(
        '-c',
        dest='cell_line',
        help='Required: Cell line for analysis[example:GM12878]',
        required=True
    )

    misc_args = parser.add_argument_group('Miscellaneous Arguments')

    misc_args.add_argument(
        '-hr',
        dest='high_res',
        help='High resolution specified[default:10kb]',
        default='10kb',
        choices=res_map.keys()
    )
    misc_args.add_argument(
        '-q',
        dest='map_quality',
        help='Mapping quality of raw data[default:MAPQGE30]',
        default='MAPQGE30',
        choices=['MAPQGE30', 'MAPQG0']
    )
    misc_args.add_argument(
        '-n',
        dest='norm_file',
        help='The normalization file for raw data[default:KRnorm]',
        default='KRnorm',
        choices=['KRnorm', 'SQRTVCnorm', 'VCnorm']
    )
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_down_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            A tools to down sample data from high resolution data.
            ----------------------------------------------------------------------
            Use example : python ./datasets/down_sample.py -hr 10kb -r 16 -c GM12878
            ----------------------------------------------------------------------
        '''
                                    ),
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c',
        dest='cell_line',
        help='Required: Cell line for analysis[example:GM12878]',
        required=True
    )
    req_args.add_argument(
        '-hr',
        dest='high_res',
        help='Required: High resolution specified[example:10kb]',
        default='10kb',
        choices=res_map.keys(),
        required=True
    )
    req_args.add_argument(
        '-r',
        dest='ratio',
        help='Required: The ratio of down sampling[example:16]',
        default=16,
        type=int,
        required=True
    )

    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_divider_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            A tools to divide data for train, predict and test.
            ----------------------------------------------------------------------------------------------------------
            Use example : python ./datasets/split.py -hr 10kb -r 16 -s train -chunk 64 -stride 64 -bound 201 -c GM12878
            ----------------------------------------------------------------------------------------------------------
        '''
                                    ),
        add_help=False
    )

    req_args = parser.add_argument_group('Required Arguments')

    req_args.add_argument(
        '-c',
        dest='cell_line',
        help='Required: Cell line for analysis[example:GM12878]',
        required=True
    )

    req_args.add_argument(
        '-hr',
        dest='high_res',
        help='Required: High resolution specified[example:10kb]',
        default='10kb',
        choices=res_map.keys(),
        required=True
    )
    req_args.add_argument(
        '-r',
        dest='ratio',
        help='Required: down_sampled ration[example:16]',
        default=16,
        required=True,
        type=int
    )
    req_args.add_argument(
        '-s',
        dest='dataset',
        help='Required: Dataset for train/valid/predict',
        default='train',
        # choices=['K562_test', 'mESC_test', 'train', 'valid', 'GM12878_test'],
    )

    method_args = parser.add_argument_group('Method Arguments')
    method_args.add_argument(
        '-chunk',
        dest='chunk',
        help='Required: chunk size for dividing[example:64]',
        default=64,
        type=int,
        required=True
    )
    method_args.add_argument(
        '-stride',
        dest='stride',
        help='Required: stride for dividing[example:64]',
        default=64,
        type=int,
        required=True
    )
    method_args.add_argument(
        '-bound',
        dest='bound',
        help='Required: distance boundary interested[example:201]',
        default=201,
        type=int,
        required=True
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def model_train_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Training the models
            --------------------------------------------------------------------------------------------
            Use example : python train.py -m HiADN -t c64_s64_train.npz -v c64_s64_valid.npz -e 50 -b 32
            --------------------------------------------------------------------------------------------
        '''
                                    ),
        add_help=False)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')

    misc_args.add_argument(
        '-m',
        dest='model',
        help='Required: models[HiADN, HiCARN, DeepHiC, HiCSR, HiCNN]',
        required=True,
        default="HiADN"
    )
    misc_args.add_argument(
        '-t',
        dest='train_file',
        help='Required: training file[example: c64_s64_train.npz]',
        required=True,
    )
    misc_args.add_argument(
        '-v',
        dest='valid_file',
        help='Required: valid file[example: c64_s64_valid.npz]',
        required=True,
    )
    misc_args.add_argument(
        '-e',
        dest='epochs',
        help='Optional: max epochs[example:50]',
        required=False,
        type=int,
        default=50
    )
    misc_args.add_argument(
        '-b',
        dest='batch_size',
        help='Optional: batch_size[example:32]',
        required=False,
        type=int,
        default=32
    )
    misc_args.add_argument(
        '-verbose',
        dest='verbose',
        help='Optional: recording in tensorboard [example:1( meaning True)]',
        required=False,
        type=int,
        default=1
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def model_predict_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Predict
            --------------------------------------------------------------------------------------------------
            Use example : python predict.py -m HiADN -t c64_s64_GM12878_test.npz -b 64 -ckpt best_ckpt.pytorch
            --------------------------------------------------------------------------------------------------
        '''
                                    ),
        add_help=False)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')

    misc_args.add_argument(
        '-m',
        dest='model',
        help='Required: models[HiADN, HiCARN, DeepHiC, HiCSR, HiCNN]',
        required=True,
        default="HiADN"
    )
    misc_args.add_argument(
        '-t',
        dest='predict_file',
        help='Required: predicting file[example: c64_s64_GM12878_test.npz]',
        required=True,
    )

    misc_args.add_argument(
        '-b',
        dest='batch_size',
        help='Optional: batch_size[example:64]',
        required=False,
        type=int,
        default=64
    )
    misc_args.add_argument(
        '-ckpt',
        dest='ckpt',
        help='Required: Checkpoint file[example:best.pytorch]',
        required=True,
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def model_visual_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Visualization
            --------------------------------------------------------------------------------------------------
            Use example : python ./visual.py -f hic_matrix.npz -s 14400 -e 14800 -p 95 -c 'Reds'
            --------------------------------------------------------------------------------------------------
        '''
                                    ),
        add_help=False)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')

    misc_args.add_argument(
        '-f',
        dest='file',
        help='Required: a npy file out from predict',
        required=True,
    )

    misc_args.add_argument(
        '-s',
        dest='start',
        help='Required: start bin[example: 14400]',
        required=True,
        type=int
    )
    misc_args.add_argument(
        '-e',
        dest='end',
        help='Required: end bin[example: 14800]',
        required=True,
        type=int
    )
    misc_args.add_argument(
        '-p',
        dest='percentile',
        help='Optional: percentile of max, the default is 95.',
        required=False,
        default=95,
        type=int
    )
    misc_args.add_argument(
        '-c',
        dest='cmap',
        help='Optional: color map[example: Reds]',
        required=False,
        type=str,
        default='Reds'
    )
    misc_args.add_argument(
        '-n',
        dest='name',
        help='Optional: the name of pic[example: chr4:14400-14800]',
        required=False,
        type=str
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def split_matrix_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                A tools to generate data for predict.
                ----------------------------------------------------------------------------------------------------------
                Use example : python ./data/split_matrix.py -chunk 64 -stride 64 -bound 201 -c GM12878
                ----------------------------------------------------------------------------------------------------------
            '''
                                    ),
        add_help=False
    )

    req_args = parser.add_argument_group('Required Arguments')

    req_args.add_argument(
        '-c',
        dest='cell_line',
        help='Required: Cell line for analysis[example:GM12878]',
        required=True
    )
    method_args = parser.add_argument_group('Method Arguments')
    method_args.add_argument(
        '-chunk',
        dest='chunk',
        help='Required: chunk size for dividing[example:64]',
        default=64,
        type=int,
        required=True
    )
    method_args.add_argument(
        '-stride',
        dest='stride',
        help='Required: stride for dividing[example:64]',
        default=64,
        type=int,
        required=True
    )
    method_args.add_argument(
        '-bound',
        dest='bound',
        help='Required: distance boundary interested[example:201]',
        default=201,
        type=int,
        required=True
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser
