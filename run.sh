#!/usr/bin/env bash

#
#for i in {1..3}
#do
#    echo "Welcome $i times"
#done

#for i in {1..10}
#    do
#      python3 train_final.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy/birds-inceptionresnetv2-final/checkpoint.pth.tar --train-batch 32 --lr 0.00045 --epochs 500 --momentum 0.0
#python3 finetune.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only True --norm rbn --resume /root/volume/cvpr/checkpoints/noisy/birds-inceptionresnetv2-rbn-ftf/model_best.pth.tar --train-batch 32 --lr 0.00045 --epochs 100 --momentum 0.0

#python3 train_final.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-inceptionresnetv2-final --train-batch 32 --lr 0.00045 --epochs 500 --momentum 0.0
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9
python3 train_noisy.py  --gpu 0,1 --pre_trained True --model inceptionresnetv2 --retrain_classifier_only False --norm rbn --resume /root/volume/cvpr/checkpoints/noisy1/birds-inceptionresnetv2-noisy/checkpoint.pth.tar --train-batch 32 --lr 0.002918 --epochs 500 --momentum 0.9


#python3 train.py  --gpu 0,1 --pre_trained False --model vgg --depth 16 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-vgg16-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 21 --momentum 0.1 --schedule 90 130
#python3 train.py  --gpu 0,1 --pre_trained False --model vgg --depth 19 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-vgg19-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#
#python3 train.py  --gpu 0,1 --pre_trained False --model resnet --depth 50 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-resnet50-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#python3 train.py  --gpu 0,1 --pre_trained False --model resnet --depth 101 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-resnet101-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#
#
#
#
#python3 train.py  --gpu 0,1 --pre_trained False --model efficientnet --depth b3 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-efficientnetB3-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#python3 train.py  --gpu 0,1 --pre_trained False --model efficientnet --depth b4 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-efficientnetB4-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#
#python3 train.py  --gpu 0,1 --pre_trained False --model efficientnet --depth b2 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-efficientnetB2-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#python3 train.py  --gpu 0,1 --pre_trained False --model vgg --depth 13 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-vgg13-rbn-bs64 --train-batch 64 --lr 0.1 --epochs 200 --momentum 0.1 --schedule 90 130
#
#python3 train.py  --gpu 0,1 --pre_trained False --model inceptionv4 --retrain_classifier_only False --norm rbn --checkpoint /root/volume/cvpr/checkpoints/noisy/birds-inceptionv4-rbn-init5-bs64 --train-batch 64 --lr 0.1 --epochs 185 --momentum 0.1 --schedule 90
#
#
##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only True --norm rbn --checkpoint checkpoints/augmentation/birds-resnet101-imInit-rbn-classifier-bs128-lr01_new --train-batch 128 --lr 0.1 --epochs 200 --momentum 0.0 --schedule 80 120 160

##python3 train.py  --gpu 0,1 --pre_trained True --retrain_classifier_only False --norm rbn --checkpoint checkpoints/augmentation/birds-resnet101-imInit-rbn-full-bs128-lr001-clr --train-batch 128 --lr 0.01 --epochs 200 --schedule 80 120 160
python3 self_ensemble.py  --gpu 0,1 --model inceptionresnetv2 --train-batch 32  --resume /root/volume/cvpr/checkpoints/noisy/birds-inceptionresnetv2-rbn-ftf/model_best.pth.tar --submission_file results/noisy/birds-InceptionResnetV2-bs32_out_ensemble_10_ftf.csv
#python3 test.py  --gpu 0 --model inceptionresnetv2