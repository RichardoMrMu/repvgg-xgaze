# -*- coding: utf-8 -*-
# @Time    : 2020-12-04 10:30
# @Author  : RichardoMu
# @File    : config.py
# @Software: PyCharm


import argparse
import yacs.config
from configs.config_node import ConfigNode
import torch
import yaml
arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--data_dir', type=str, default='/home/data/tbw_data/xgaze',
                      help='Directory of the data')
data_arg.add_argument('--batch_size', type=int, default=200,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=5,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=25,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.0001,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--lr_decay_factor', type=int, default=0.1,
                       help='Number of epochs to wait before reducing lr')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--pre_trained_model_path', type=str, default='./ckpt/epoch_24_ckpt.pth.tar',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--print_freq', type=int, default=1000,
                      help='How frequently to print training details')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/exp00',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--default_model',type=str,default='resnet18',
                      help="choose network including resnet18, resnet34, resnet50, resnet101 etc.")
misc_arg.add_argument("--dataset_split",type=list,default=[0.85,0.1,0.05],
                      help='choose train val test dataset split')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def load_config() -> yacs.config.CfgNode:
    parse = argparse.ArgumentParser()
    parse.add_argument('--config',default='./configs/repvgg_d2se_train.yaml',type=str)
    args = parse.parse_args()
    with open(args.config,'r',encoding='utf-8') as f:

        results = yaml.load(f.read())
    config = ConfigNode(results)
    # config.merge_from_file(args.config)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.train_dataloader.pin_memory = False
        config.train.val_dataloader.pin_memory = False
        config.test.dataloader.pin_memory = False
    return config
