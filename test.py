#-*- coding:utf-8 -*-
import os
from dataloader.data import get_indist_data
import torch
from utils.args import args_for_train_tl

from noisystudent.ensemble import create_ensemble_noisy_labels

# get all the input variables
args = args_for_train_tl()
# ToDo: load args from experiemnt_ags file from args.resume
# https://stackoverflow.com/questions/28348117/using-argparse-and-json-together
state = {k: v for k, v in args._get_kwargs()}

args.model = 'inceptionresnetv2'
args.resume = '/root/volume/cvpr/checkpoints/inception/birds-inceptionresnetv2-rbn-bs32/checkpoint.pth.tar'

############################################
# Create and Load Model
############################################
from models.models import get_model
model, image_size = get_model(args, num_classes=200, use_pretrained=False, train=False)
model = model.cuda()
model = torch.nn.DataParallel(model)

test_annots = "anno_u_train_in.json"
test_folder = "u_train_in"
bs = 64
indist_loader = get_indist_data(annot_file=test_annots, in_dist_path=test_folder, image_size=image_size, batch_size=bs)

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

print('==> Creating Noisy Labels..')
create_ensemble_noisy_labels(model, indist_loader, "results/noisy_labels/inceptionresnetv2_rbn_bs32_indist_soft_en10.csv",
                             ensemble_epochs=10, tolerance=0.0, test_flag=False)


