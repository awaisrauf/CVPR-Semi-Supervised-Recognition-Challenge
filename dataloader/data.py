import json
import os

import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from .data_utils import load_trainval_annotations
from .datasets import TrainDataset, TestDataset, NoisyStudentDataset

HEIGHT = 224
WIDTH = 224
N_CLASSES = 200


def get_transforms(data, normalize=None, image_size=224):
    assert data in ('train', 'valid', 'test')

    if normalize == "advprop":  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    max_crop = image_size // 5
    full_train_transform = A.Compose([
        A.RandomResizedCrop(image_size, image_size),
        A.OneOf([
            A.RandomRain(p=0.1),
            A.GaussNoise(mean=15, p=0.5),
            A.GaussianBlur(blur_limit=10, p=0.4),
            A.MotionBlur(p=0.2)
            ], p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        # A.OneOf([
        # 	A.RGBShift(p=1.0,
        # 		r_shift_limit=(-10, 10),
        # 		g_shift_limit=(-10, 10),
        # 		b_shift_limit=(-10, 10)),
        # 	A.RandomBrightnessContrast(	brightness_limit=0.3, contrast_limit=0.1, p=1),
        # 	A.HueSaturationValue(hue_shift_limit=20, p=1),
        # 	], p=0.6),
        A.Cutout(num_holes=1, max_h_size=max_crop, max_w_size=max_crop, p=0.3),
        A.OneOf([
            A.Flip(),
            A.Rotate()
            ], p=0.5),
        normalize,
        ToTensorV2()])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if data == 'train':
        return full_train_transform

    elif data == "valid":
        return valid_transform

    elif data == "test":
        return test_transform


def get_trainval_data(batch_size=128):
    train_path = "/root/volume/birds/trainval_images"
    val_path = "/root/volume/birds/val"
    train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")

    train_dataset = TrainDataset(train_df, train_path, transform=get_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_dataset = TrainDataset(valid_df, val_path, transform=get_transforms('valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)
    return train_loader, valid_loader


def get_final_train_data(batch_size=128):
    train_path = "/root/volume/birds/trainval_images"
    train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")
    # concat two dataframes
    frames = [train_df, valid_df]
    final_df = pd.concat(frames)

    train_dataset = TrainDataset(final_df, train_path, transform=get_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    return train_loader


def get_test_data():
    batch_size = 128
    base_path = "/root/volume/birds"
    test_path = os.path.join(base_path, "test")
    # get annotations
    test_annot_file = "/root/volume/birds/annotation/annotation/anno_test.json"
    with open(os.path.join(test_annot_file), "r", encoding="ISO-8859-1") as file:
        test = json.load(file)
    test_df = pd.DataFrame(test['images'])

    test_dataset = TestDataset(test_df, test_path, transform=get_transforms(data='test'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    return test_loader


def get_noisy_student_data(noisy_annots_path):
    batch_size1 = 16
    train_path = "/root/volume/birds/trainval_images"
    val_path = "/root/volume/birds/val"
    train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")

    train_dataset = TrainDataset(train_df, train_path, transform=get_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True, pin_memory=True, num_workers=2)

    valid_dataset = TrainDataset(valid_df, val_path, transform=get_transforms('valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)

    batch_size2 = 112
    in_dist_data_path = '/root/volume/birds/u_train_out/u_train_out'
    noisy_df = pd.load_csv(noisy_annots_path)
    noisy_dataset = NoisyStudentDataset(noisy_df, in_dist_data_path, transform=get_transforms('train'))
    noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size2, shuffle=True, pin_memory=True, num_workers=2)

    concate_dataset = torch.utils.data.ConcatDataset([train_loader, noisy_loader])
    return concate_dataset, valid_loader


if __name__ == "__main__":
    train = get_final_train_data(batch_size=1)

    f_train = get_final_train_data(batch_size=1)
    # print(len(train), len(val), len(f_train))
    print(len(train))
    for img, label in train:
        print(img.shape)
        break
# for img, label in train:
# 	print(img, label)
