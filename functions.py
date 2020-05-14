import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.optim as optim
from utils import Bar, AverageMeter, accuracy
from PIL import ImageFile
import utils
import numpy as np
import pandas as pd
import torch.nn as nn
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_one_epoch(trainloader, model, criterion, optimizer, use_cuda=True, test_flag=False):
    """
    trains the model on trainloader data for one epoch
    :param trainloader: pytroch dataloader that gives one batch of data at each iteration
    :type trainloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param optimizer: gradient descent based optimzer to update model weights
    :type optimizer: pytorch class
    :param use_cuda: flag to set to use cuda or not
    :type use_cuda: boolean
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: floats
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # for test case only
        if test_flag is True and batch_idx > 1:
            break
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        #
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: ' \
                     '{loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(batch=batch_idx + 1,
                                                                                 size=len(trainloader),
                                                                                 data=data_time.avg,bt=batch_time.avg,
                                                                                 total=bar.elapsed_td, eta=bar.eta_td,
                                                                                 loss=losses.avg, top1=top1.avg,
                                                                                 top5=top5.avg,)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, top5.avg


def test(testloader, model, criterion, use_cuda=True, test_flag=False):
    """
    checks performance of the model
    :param testloader: pytroch dataloader that gives one batch of data at each iteration
    :type testloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param use_cuda: whether to put inputs, targets on cuda or not
    :type use_cuda: boolean
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: floats
    """
    # initialization of telemetery
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # testing
    model.eval()   # switch to test mode
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # for test case only
        if test_flag and batch_idx > 1:
            break

        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |' \
                      ' Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                                                                                                batch=batch_idx + 1,
                                                                                                size=len(testloader),
                                                                                                data=data_time.avg,
                                                                                                bt=batch_time.avg,
                                                                                                total=bar.elapsed_td,
                                                                                                eta=bar.eta_td,
                                                                                                loss=losses.avg,
                                                                                                top1=top1.avg,
                                                                                                top5=top5.avg)
        bar.next()
    bar.finish()

    return losses.avg, top1.avg, top5.avg


def set_optimizer(model, args):
    """
    sets optimizer according to ars
    :param model: deep learning model
    :type model: pytroch class
    :param args: arguments passed
    :type args:
    :return:
    :rtype:
    """
    params = [{'params': [p for p in model.parameters() if not getattr(p, 'bin_gate', False)]},
              {'params': [p for p in model.parameters() if getattr(p, 'bin_gate', False)],
               'lr': args.lr * args.bin_lr, 'weight_decay': 0}]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(state, optimizer, epoch, gamma, schedule):
    if epoch in schedule:
        state *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return state


def make_final_submission(Idxs, preds, submission_file_path):
    final_pred1 = np.argsort(preds, axis=1)[:, ::-1]
    final_pred = final_pred1[:, 0:5]
    sub = []
    for num, item in enumerate(final_pred):
        lbs = ""
        for i in item:
            lbs+=str(i)+" "
        sub.append([Idxs[num], lbs])

    pd_sub = pd.DataFrame(sub, columns=["Id", "Category"])
    pd_sub.to_csv(submission_file_path, index=False)



def log_tensorboard(writer, n_iter, train_loss, train_acc, train_acc5,val_loss, val_acc, val_acc5):
    writer.add_scalar('train_loss', train_loss, n_iter)
    writer.add_scalar('val_loss', val_loss, n_iter)
    writer.add_scalar('train_acc', train_acc, n_iter)
    writer.add_scalar('val_acc', val_acc, n_iter)
    writer.add_scalar('train_acc5', train_acc5, n_iter)
    writer.add_scalar('val_acc5', val_acc5, n_iter)

