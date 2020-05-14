"""
"""
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import Logger, mkdir_p
from functions import train_one_epoch, test, set_optimizer, save_checkpoint,\
    adjust_learning_rate, log_tensorboard
from utils.args import args_for_train_noisy
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim
from dataloader.data_utils import get_class_weights


# get all the input variables
args = args_for_train_noisy()
state = {k: v for k, v in args._get_kwargs()}
if not os.path.isdir(args.checkpoint):
    print("==> Creating checkpoint folder")
    mkdir_p(args.checkpoint)
# save all the arguments
args_save_path=args.checkpoint if args.resume=='' else os.path.dirname(args.resume)
with open(os.path.join(args_save_path, 'experiment_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
writer = SummaryWriter(os.path.join(args_save_path, "tensorboard"))
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# set parameters
best_acc = 0  # best test accuracy
do_save_checkpoint = True
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


from dataloader.data_utils import get_class_num
def get_class_weights(beta):
    samples_per_cls = get_class_num()
    cls_num_list = samples_per_cls/np.sum(samples_per_cls)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return cls_num_list, per_cls_weights

############################################
# Create and Load Model
############################################
from models.model1 import get_model
model = get_model(num_classes=200, norm=args.norm, pre_trained=args.pre_trained, train_classifier_only=args.retrain_classifier_only)
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
optimizer = set_optimizer(model, args)


##############################################
# Load Dataset
##############################################
print('==> Preparing dataset')
from dataloader.data import get_trainval_data, get_final_train_data
trainloader, validloader = get_trainval_data(batch_size=args.train_batch)



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

#############################################
# Train and Validate
#############################################
betas = [0.9999, 0.9999]
for epoch in range(start_epoch, args.epochs-20):
    # Deferred Re-balancing Optimization Schedule (https://arxiv.org/pdf/1906.07413.pdf)
    if epoch<3600:
        beta_idx = epoch // 160
        cls_num_list, per_cls_weights = get_class_weights(betas[beta_idx])
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
    else:
        beta_idx = epoch//160
        cls_num_list, per_cls_weights = get_class_weights(betas[beta_idx])
        criterion = LDAMLoss(cls_num_list=cls_num_list, weight=per_cls_weights).cuda()
    state['lr'] = adjust_learning_rate(state['lr'], optimizer, epoch, args.gamma, args.schedule)
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    train_loss, train_acc, train_acc5 = train_one_epoch(trainloader, model, criterion, optimizer, use_cuda=use_cuda)
    valid_loss, valid_acc, valid_acc5 = test(validloader, model, criterion, use_cuda=use_cuda)
    logger.append([state['lr'], train_loss, valid_loss, train_acc, valid_acc, train_acc5, valid_acc5])
    # log_tensorboard(writer, epoch, train_loss, train_acc, train_acc5, valid_loss, valid_acc, valid_acc5)

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
for fine_epoch in range(args.epochs-20, args.epochs):
    # Deferred Re-balancing Optimization Schedule (https://arxiv.org/pdf/1906.07413.pdf)
    cls_num_list, per_cls_weights = get_class_weights(0.9999)
    criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
    state['lr'] = args.lr
    print('\nEpoch: [%d | %d] LR: %f' % (fine_epoch + 1, args.epochs, state['lr']))
    train_loss, train_acc, train_acc5 = train_one_epoch(validloader, model, criterion, optimizer, use_cuda=use_cuda)
    valid_loss, valid_acc, valid_acc5 = train_loss, train_acc, train_acc5#test(validloader, model, criterion, use_cuda=use_cuda)
    logger.append([state['lr'], train_loss, valid_loss, train_acc, valid_acc, train_acc5, valid_acc5])
    # log_tensorboard(writer, fine_epoch, train_loss, train_acc, train_acc5, valid_loss, valid_acc, valid_acc5)

    # save model ap
    is_best = valid_acc > best_acc
    best_loss = max(valid_acc, best_acc)
    best_acc = max(valid_acc, best_acc)
    if do_save_checkpoint:
        save_checkpoint({
                'epoch': fine_epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)


logger.close()
print('Best acc:', best_acc)

