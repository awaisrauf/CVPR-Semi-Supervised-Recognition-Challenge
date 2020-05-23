"""
"""
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import Logger, mkdir_p
from functions import train_one_epoch, test, set_optimizer, save_checkpoint,\
    adjust_learning_rate, log_tensorboard
from utils.args import args_for_train_tl
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim
from dataloader.data_utils import get_class_weights
from models.loss import WL2SP

# get all the input variables
args = args_for_train_tl()
state = {k: v for k, v in args._get_kwargs()}
if not os.path.isdir(args.checkpoint):
    print("==> Creating checkpoint folder")
    mkdir_p(args.checkpoint)
# save all the arguments
args_save_path=args.checkpoint if args.resume=='' else os.path.dirname(args.resume)
with open(os.path.join(args_save_path, 'experiment_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
# writer = SummaryWriter(os.path.join(args_save_path, "tensorboard"))
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# set parameters
best_acc = 0  # best test accuracy
do_save_checkpoint = True
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


############################################
# Create and Load Model
############################################
from models.models import get_model
model, image_size = get_model(args, num_classes=200, use_pretrained=False, train=True)
image_size=image_size
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
optimizer = set_optimizer(model, args)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, weight_decay=0.0, momentum=0.0, centered=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94, last_epoch=-1)

# pre-trained weights for loss function
pretrained_weights = {}
for n, m in model.named_modules():
    if "conv" in n:
        try:
            pretrained_weights[n] = m.weight
        except:
            pass

cls_num_list, per_cls_weights = get_class_weights(0.99)
criterion = WL2SP(class_weights=per_cls_weights, pretrained_weights=pretrained_weights)


##############################################
# Load Dataset
##############################################
print('==> Preparing dataset')
from dataloader.data import get_trainval_data, get_final_train_data
trainloader, validloader = get_trainval_data(batch_size=args.train_batch, val_batch_size=64, image_size=image_size)


##############################################
# Initialize Model
#############################################
title = '{}-{}-{}-{}'.format(args.dataset,args.model, args.depth, args.norm)
# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
else:
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train Acc.5',
                      'Valid Acc.5'])

############################################
# Train and Validate
############################################
base_lr = 0.0045
lrsL = [i*base_lr for i in range(1, 5)]
lrsS = [ base_lr/i for i in range(1, 5)]
valid_acc_3_epochs = []
for epoch in range(start_epoch, args.epochs-20):

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, scheduler.get_lr()))
    train_loss, train_acc, train_acc5 = train_one_epoch(trainloader, model, criterion, optimizer, use_cuda=use_cuda)
    valid_loss, valid_acc, valid_acc5 = test(validloader, model, criterion, use_cuda=use_cuda)
    logger.append([state['lr'], train_loss, valid_loss, train_acc, valid_acc, train_acc5, valid_acc5])

    # if epoch%4==0:
    #     scheduler.step()
    # save model ap
    is_best = valid_acc > best_acc
    best_loss = max(valid_acc, best_acc)
    best_acc = max(valid_acc, best_acc)
    if do_save_checkpoint:
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

###################################
# FineTune on Validation Data ###
###################################
# print("==> Loading best model for fine tuning...")
# checkpoint = torch.load(os.path.join(args.checkpoint, "model_best.pth.tar"))
# model.load_state_dict(checkpoint['state_dict'])
#
# # freeze feature extraction layers
# for n, m in model.named_modules():
#     if "conv" in n:
#         m.requires_grad = False
#
#
#
# print("==> Fine Tuning on Validation Data")
# for fine_epoch in range(args.epochs-20, args.epochs):
#     # Deferred Re-balancing Optimization Schedule (https://arxiv.org/pdf/1906.07413.pdf)
#     cls_num_list, per_cls_weights = get_class_weights(0)
#     criterion =  WL2SP(class_weights=per_cls_weights, pretrained_weights=pretrained_weights)
#     state['lr'] = args.lr
#     print('\nEpoch: [%d | %d] LR: %f' % (fine_epoch + 1, args.epochs, state['lr']))
#     train_loss, train_acc, train_acc5 = train_one_epoch(validloader, model, criterion, optimizer, use_cuda=use_cuda)
#     valid_loss, valid_acc, valid_acc5 = train_loss, train_acc, train_acc5#test(validloader, model, criterion, use_cuda=use_cuda)
#     logger.append([state['lr'], train_loss, valid_loss, train_acc, valid_acc, train_acc5, valid_acc5])
#     # log_tensorboard(writer, fine_epoch, train_loss, train_acc, train_acc5, valid_loss, valid_acc, valid_acc5)
#
#     # save model ap
#     is_best = valid_acc > best_acc
#     best_loss = max(valid_acc, best_acc)
#     best_acc = max(valid_acc, best_acc)
#     if do_save_checkpoint:
#         save_checkpoint({
#                 'epoch': fine_epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'acc': valid_acc,
#                 'best_acc': best_acc,
#                 'optimizer' : optimizer.state_dict(),
#             }, is_best, checkpoint=args.checkpoint)


logger.close()
print('Best acc:', best_acc)

