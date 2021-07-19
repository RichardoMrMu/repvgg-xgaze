# -*- coding: utf-8 -*-
# @Time    : 2020-12-07 14:23
# @Author  : RichardoMu
# @File    : trainer_tbw.py
# @Software: PyCharm


import torch
import torch.nn.functional as F
import time
import numpy as np
from utils import AverageMeter,angular_error

def train(epoch,model,optimizer,scheduler,train_loader,
          config,logger=None
          ,tensorboard_writer=None
          ):
    """
    Train the model for 1 epoch of the training set.
    """
    num_train = len(train_loader.dataset)
    # print(f"\n[*] Train on {num_train} samples")
    # print(
    #     '\nEpoch: {}/{} - base LR: {:.6f}'.format(
    #         epoch + 1, config.scheduler.epochs, config.optimizer.init_lr)
    # )
    logger.info(f'[*] Train on {num_train} samples\n'
                f'Epoch: {epoch+1}/{config.scheduler.epochs} - base LR: {config.optimizer.init_lr:.6f} ')
    # for param_group in optimizer.param_groups:
    #     print("Learning rate:",param_group['lr'])
    # train for 1 epoch
    logger.info(f"Now going to train")
    model.train()

    batch_time = AverageMeter()
    errors = AverageMeter()
    losses_gaze = AverageMeter()

    tic = time.time()
    train_step = 0
    for i, (input_img,target_var) in enumerate(train_loader):
        # input_var = torch.autograd.Variable(input_img.float().cuda())
        # target_var = torch.autograd.Variable(target_var.float().cuda())
        input_var = input_img.cuda(non_blocking=True)
        target_var = target_var.cuda(non_blocking=True).float()
        # train gaze net
        pred_gaze = model(input_var)

        gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(),target_var.cpu().data.numpy()))
        errors.update(gaze_error_batch.item(),input_var.size()[0])
        # 使用 l1 loss
        loss_gaze = F.l1_loss(pred_gaze,target_var)


        optimizer.zero_grad()
        loss_gaze.backward()
        optimizer.step()
        losses_gaze.update(loss_gaze.item(),input_var.size()[0])
        
        # if i % config.print_freq == 0:
        #     tensorboard_writer.add_scalar('Loss/gaze',losses_gaze.avg,train_step)
        # report information
        if i % config.train.print_freq == 0 and i is not 0:

            toc = time.time()
            batch_time.update(toc-tic)
            tic = time.time()
            # estiamte the finish time
            est_time = (config.scheduler.epochs-epoch) * ((num_train) /(config.train.batch_size*config.train.print_freq)) * batch_time.avg / 60.0
            logger.info(
                f"Epoch: {epoch+1}/{config.scheduler.epochs}\n"
                f"Estimated training time left:{np.round(est_time)} mins \n"
                f'--------------------------------------------------------------------\n'
                f"train error: {errors.avg:.3f} - loss_gaze: {losses_gaze.avg:.5f}\n"
                f"iteration:{train_step}\n"
                f"current learning rate:{scheduler.get_last_lr()[0]}\n"
                f'Current batch running time is {np.round(batch_time.avg / 60.0)} mins\n'
                f'Estimated training time left: {np.round(est_time)} mins\n'
            )
            tensorboard_writer.add_scalar('training loss',losses_gaze.avg,len(train_loader)*epoch/50000+i/500)
            tensorboard_writer.add_scalar('training error',errors.avg,len(train_loader)*epoch/50000+i/500)
            errors.reset()
            losses_gaze.reset()
      

        train_step += 1   
    scheduler.step() # update learning rate   
    toc = time.time()
    batch_time.update(toc-tic)
    # print("running time is :",batch_time.avg)
    logger.info(
        f'running time is:{batch_time.avg}'
    )
    


def validate(epoch,model,test_loader,config,logger=None,
             tensorboard_writer=None
             ):
    model.eval()
    errors = AverageMeter()
    losses_gaze = AverageMeter()
    tic = time.time()
    with torch.no_grad():
        for step, (input_var,target_var) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input_var.float().cuda())
            target_var = torch.autograd.Variable(target_var.float().cuda())
            # test gaze net
            pred_gaze = model(input_var)
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(),input_var.size()[0])

            loss_gaze = F.l1_loss(pred_gaze,target_var)
            losses_gaze.update(loss_gaze.item(),input_var.size()[0])
            tensorboard_writer.add_scalar('test loss', losses_gaze.avg,len(test_loader)/config.test.batch_size*epoch + step)
            tensorboard_writer.add_scalar('test error', errors.avg, len(test_loader)/config.test.batch_size*epoch + step)
    toc = time.time()
    total_time = (toc - tic) /60.0
    # print(f"Epoch : {epoch}, error : {errors.avg}, loss : {losses_gaze.avg}, total time : {total_time},current learning rate:{optimizer.get_lr()[0]}")
    logger.info(
        f'let us now do the val/test'
        f"Epoch : {epoch}, error : {errors.avg}, loss : {losses_gaze.avg}, total time : {total_time}"
    )
    return errors.avg


def save_checkpoint(state, add=None):
    """
    Save a copy of the model
    """
    if add is not None:
        ckpt_path = add + '_ckpt.pth.tar'
    else:
        ckpt_path ='ckpt.pth.tar'
    torch.save(state, ckpt_path)

    print('save file to: ', ckpt_path)
