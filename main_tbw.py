# -*- coding: utf-8 -*-
# @Time    : 2020-12-07 11:18
# @Author  : RichardoMu
# @File    : main_tbw.py
# @Software: PyCharm
import torch
from config import load_config
from data_loader_tbw import get_data_loader
from trainer_tbw import train,validate,save_checkpoint
from utils import sgd_optimizer
import numpy as np
import os
from models import get_RepVGG_func_by_name
from logger import create_logger
import pathlib
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from tensorboardX import SummaryWriter


def run(config):
    if config.train.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = config.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.cudnn.benchmark
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # kwargs = {'num_workers': config.num_workers, 'is_shuffle': config.shuffle}

    # configure logging
    log_dir = config.log.log_dir
    # if os.path.exists(log_dir) and os.path.isdir(log_dir):
    #     shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    output_dir = pathlib.Path(log_dir)
    time_now = datetime.datetime.now()
    str_now = datetime.datetime.strftime(time_now, '%Y-%m-%d-%H-%M-%S')
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename=str_now+'log.txt')
    logger.info(config)

    # instantiate data loaders
    train_loader,val_loader,test_loader = get_data_loader(config)
    # lr = config.optimizer.init_lr 
    # lr_patience = config.train.lr_patience
    # lr_decay_factor = config.train.lr_decay_factor
    use_gpu = config.train.use_gpu
    ckpt_dir = config.train.ckpt_dir  # output dir
    # summary writer
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    writer = SummaryWriter(ckpt_dir)
    if use_gpu and torch.cuda.device_count() > 1:
        logger.info(
            f'Let us use  {torch.cuda.device_count()} GPUs!'
        )

    # build model
    repvgg_build_func = get_RepVGG_func_by_name(config.model.name)
    model = repvgg_build_func(deploy=False)
    # load model
    if os.path.isfile(config.model.weights):
        logger.info(f'=> loading chechpoints {config.model.weights}')
        checkpoints = torch.load(config.model.weights)
        if 'state_dict' in checkpoints:
            checkpoints = checkpoints['state_dict']
        ckpt = {k.replace('module.',''):v for k,v in checkpoints.items()} # strip the names
        layer_names =list(model.state_dict().keys())
        # print(model.state_dict()[layer_names[2]])
        model.load_state_dict(ckpt,strict=False)
        # print(model.state_dict()[layer_names[2]])
    else:
        print(f"=> no checkpoint found at {config.model.weights}")
    # model = gaze_network(net_choice=config.model)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        # use parallel gpus to train model
        model = torch.nn.DataParallel(model).cuda()

    # get number of parameters of the model
    print('[*] Number of model parameters: {:,}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    logger.info(
        f'[*] Number of model parameters: {sum([p.data.nelement() for p in model.parameters()]):,}'
    )
    # initialize optimizer and scheduler
    # optimizer = torch.optim.Adam(
    #     model.parameters(),lr=lr
    # )
    # optimizer = sgd_optimizer(model,config.optimizer.init_lr,config.optimizer.momentum,config.optimizer.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.optimizer.init_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=config.scheduler.epochs*len(train_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=config.optimizer.lr_patience,gamma=config.optimizer.lr_decay_factor)
    if config.train.resume:
        if os.path.isfile(config.train.resume):
            logger.info(f"=> loading checkpoint {config.train.resume}")
            # loc = 'cuda:{}'.format(0,1,2,3)
            # checkpoint = torch.load(config.train.resume,map_location=loc)
            checkpoint = torch.load(config.train.resume)
            config.scheduler.start_epoch = checkpoint['epoch']
            # best_error = 2.905
            best_error = checkpoint['best_error']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
            scheduler.load_state_dict(checkpoint['schedule_state'])
            logger.info(f'=> loaded checkpoint {config.train.resume} (epoch :{config.scheduler.start_epoch}')
    if config.scheduler.start_epoch == 0:
        best_error = 0
        best_epoch = 0
    for epoch in range(config.scheduler.start_epoch,config.scheduler.epochs):
        train(epoch,model=model,optimizer=optimizer,
              scheduler=scheduler,train_loader=train_loader,
              config=config,logger=logger,tensorboard_writer=writer)
        logger.info(
            f'begin validation'
        )
        val_error = validate(epoch,model=model,test_loader=val_loader,config=config,
                             # tensorboard_writer=writer
                             logger=logger,tensorboard_writer=writer
                             )
        logger.info(
            f'begin test'
        )
        test_error = validate(epoch,model=model,test_loader=test_loader,config=config,
                             # tensorboard_writer=writer
                             logger=logger,tensorboard_writer=writer
                              )
        if epoch == 0:
            best_error = val_error
            best_epoch = epoch
        if val_error < best_error:
            best_epoch = epoch
            best_error = val_error
            save_checkpoint({
                'epoch':epoch + 1,
                'arch': config.model.name,
                'model_state':model.state_dict(),
                'best_error': best_error,
                'optim_state':optimizer.state_dict(),
                'schedule_state':scheduler.state_dict()
            },
                add= os.path.join(ckpt_dir,str(config.model.name) + "_epoch_" + str(epoch))
            )
    print(f"best error : {best_error}, best epoch : {best_epoch}")
    logger.info(f"best error : {best_error}, best epoch : {best_epoch}")



if __name__ == '__main__':
    # config, unparsed = get_config()
    config = load_config()
    run(config)
