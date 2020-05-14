#!/usr/bin/env bash

##python3 train.py  --gpu 0,1 --pre_trained False --norm bn --checkpoint checkpoints/birds-resnet50-scratch-bn-lr001 --train-batch 128 --lr 0.01 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained True --norm bn  --checkpoint checkpoints/birds-resnet50-imInit-bn-lr001 --train-batch 128 --lr 0.01 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained False --norm rbn --checkpoint checkpoints/birds-resnet50-scratch-rbn-lr01-bs32 --train-batch 32 --lr 0.1 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/birds-resnet50-imInit-rbn-lr01-bs32 --train-batch 32 --lr 0.1 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm bn --checkpoint checkpoints/birds-resnet50-imInit-bn-lr01-bs32 --train-batch 32 --lr 0.1 --epochs 100
##
##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/birds-resnet50-imInit-rbn-lr001-bs32 --train-batch 32 --lr 0.01 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained False --retrain_classifier_only False --norm rbn --checkpoint checkpoints/birds-resnet50-scratch-rbn-lr001-bs32 --train-batch 32 --lr 0.01 --epochs 100
##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/birds-resnet50-imInit-bn-lr001-bs32 --train-batch 32 --lr 0.01 --epochs 100
#
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-classifier-bs128-lw-lr0001 --train-batch 128 --lr 0.001 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-classifier-bs128-lr0001 --train-batch 128 --lr 0.001 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-full-bs128-lw-lr0001 --train-batch 128 --lr 0.001 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs128-lr0001 --train-batch 128 --lr 0.001 --epochs 100
#
#
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-classifier-bs128-lw-lr001 --train-batch 128 --lr 0.01 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-classifier-bs128-lr001 --train-batch 128 --lr 0.01 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-full-bs128-lw-lr001 --train-batch 128 --lr 0.01 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs128-lr001 --train-batch 128 --lr 0.01 --epochs 100
#
#
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-classifier-bs128-lw-lr01 --train-batch 128 --lr 0.1 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-classifier-bs128-lr01 --train-batch 128 --lr 0.1 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-full-bs128-lw-lr01 --train-batch 128 --lr 0.1 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs128-lr01 --train-batch 128 --lr 0.1 --epochs 100
#
#
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-effB4-imInit-rbn-full-bs64 --train-batch 64 --lr 0.1 --epochs 200 --schedule 50 100 150
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs32-lr01 --train-batch 32 --lr 0.1 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-full-bs32-lw-lr001 --train-batch 32 --lr 0.01 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs32-lr001 --train-batch 32 --lr 0.01 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/new/birds-resnet50-imInit-rbn-full-bs32-lw-lr0001 --train-batch 32 --lr 0.001 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-imInit-bn-full-bs32-lr0001 --train-batch 32 --lr 0.001 --epochs 100
#
#
#
#python3 train.py  --gpu 0,1 --pre_trained False --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-Scratch-bn-classifier-bs128-lr01 --train-batch 128 --lr 0.1 --epochs 100
#python3 train.py  --gpu 0,1 --pre_trained False --retrain_classifier_only False --norm bn --checkpoint checkpoints/new/birds-resnet50-Scratch-bn-classifier-bs32-lr01 --train-batch 32 --lr 0.1 --epochs 100

#pip install pretrainedmodelsh 128
python3 train.py  --gpu 0,1 --pre_trained True --model resnet --depth 18 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-resnet18-rbn --train-batch 128 --lr 0.01 --epochs 150 --momentum 0.0 --schedule 60 100
python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/augmentation/birds-resnet101-imInit-rbn-classifier-bs128-lr01_new --train-batch 128 --lr 0.1 --epochs 200 --momentum 0.0 --schedule 80 120 160

#python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/augmentation/birds-resnet101-imInit-rbn-full-bs128-lr001-clr --train-batch 128 --lr 0.01 --epochs 200 --schedule 80 120 160
python3 submit.py  --gpu 0  --resume /root/volume/cvpr/checkpoints/abc/birds-resnet101-bn-drw-m0/model_best.pth.tar --submission_file results/sub/birds-resnet101-bn-m0-finetuned.csv
#python3 submit.py  --gpu 0  --resume checkpoints/augmentation/birds-resnet101-imInit-rbn-full-bs128-lr01-clr/model_best.pth.tar --submission_file results/submission/resnet101-best-augmentation-lr01-clr.csv