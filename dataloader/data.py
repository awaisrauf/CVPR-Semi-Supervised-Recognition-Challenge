import json
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms

from .data_utils import load_trainval_annotations
from .datasets import TrainDataset, TestDataset, NoisyStudentDataset, InDistributionDataset, OutDistributionDataset

HEIGHT = 224
WIDTH = 224
N_CLASSES = 200


def get_transforms(data, normalize=None, image_size=224):
	# assert data in ('train', 'valid', 'test', 'indist')

	if normalize == "advprop":  # for models using advprop pretrained weights
		normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
	else:
		normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val = np.random.uniform(0.6, 1.4)
	max_crop = image_size // 5
	full_train_transform = A.Compose([
		A.RandomResizedCrop(image_size, image_size),
		A.HorizontalFlip(0.5),
		A.HueSaturationValue(val),
		A.RandomBrightness(val),
		A.FancyPCA(alpha=0.01),
		A.OneOf([
			A.RandomRain(p=0.1),
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
		# 	           r_shift_limit=(-10, 10),
		# 	           g_shift_limit=(-10, 10),
		# 	           b_shift_limit=(-10, 10)),
		# 	A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, p=1),
		# 	A.HueSaturationValue(hue_shift_limit=20, p=1),
		# ], p=0.2),
		A.Cutout(num_holes=1, max_h_size=max_crop, max_w_size=max_crop, p=0.3),

		normalize,
		ToTensorV2()])

	valid_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((256, 256)),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	test_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((300, 300)),
		transforms.RandomResizedCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	test_ensemble_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((300, 300)),
		transforms.RandomResizedCrop(image_size),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	])
	indist_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((256, 256)),
		transforms.RandomResizedCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	outdist_transform = transforms.Compose([
		transforms.ToPILImage(),
		# transforms.Resize((256, 256)),
		transforms.RandomResizedCrop((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	if data == 'train':
		return full_train_transform

	elif data == "valid":
		return valid_transform

	elif data == "test":
		return test_transform

	elif data == "test_ensemble":
		return test_ensemble_transform


	elif data == "indist":
		return indist_transform

	elif data == "outdist":
		return outdist_transform


def get_trainval_data(batch_size=128, val_batch_size=128, image_size=224):
	train_path = "/root/volume/birds/trainval_images"
	val_path = "/root/volume/birds/val"
	train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")

	train_dataset = TrainDataset(train_df, train_path, transform=get_transforms('train', image_size=image_size))
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

	valid_dataset = TrainDataset(valid_df, val_path, transform=get_transforms('valid', image_size=image_size))
	valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=2)
	return train_loader, valid_loader


def get_final_train_data(batch_size=128, image_size=224):
	train_path = "/root/volume/birds/trainval_images"
	train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")
	# concat two dataframes
	frames = [train_df, valid_df]
	final_df = pd.concat(frames)

	train_dataset = TrainDataset(final_df, train_path, transform=get_transforms('train', image_size=image_size))
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

	return train_loader


def get_test_data(batch_size=128, image_size=224, transform_type="test"):
	base_path = "/root/volume/birds"
	test_path = os.path.join(base_path, "test")
	# get annotations
	test_annot_file = "/root/volume/birds/annotation/annotation/anno_test.json"
	with open(os.path.join(test_annot_file), "r", encoding="ISO-8859-1") as file:
		test = json.load(file)
	test_df = pd.DataFrame(test['images'])

	test_dataset = TestDataset(test_df, test_path, transform=get_transforms(data=transform_type, image_size=image_size))
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
	return test_loader


def get_noisy_student_data(noisy_data_folder, noisy_annots_path, batch_size=32, image_size=224):
	base_path = "/root/volume/birds/"
	# train_path = os.path.join(base_path, "trainval_images")
	# train_df, valid_df = load_trainval_annotations(base_path="/root/volume/birds/annotation/annotation")
	# train_dataset = TrainDataset(train_df, train_path, transform=get_transforms('train', image_size=image_size))

	noisy_data_path = os.path.join(base_path, noisy_data_folder)
	noisy_df = pd.read_csv(noisy_annots_path)
	noisy_dataset = NoisyStudentDataset(noisy_df, noisy_data_path, transform=get_transforms('indist', image_size=image_size))
	noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
	# DataLoader(nois, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
	# concate_dataset = torch.utils.data.ConcatDataset([noisy_dataset, train_dataset])
	# concat_loader = DataLoader(concate_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
	return noisy_loader


def get_indist_data(annot_file='anno_u_train_in.json', in_dist_path="u_train_in", image_size=224, batch_size=64):
	base_path = "/root/volume/birds"
	in_dist_data_path = os.path.join(base_path, in_dist_path)
	# get annotations
	annot_path = os.path.join("/root/volume/birds/annotation/annotation/", annot_file)
	with open(os.path.join(annot_path), "r", encoding="ISO-8859-1") as file:
		indist = json.load(file)
	indist_df = pd.DataFrame(indist['images'])
	indist_dataset = InDistributionDataset(indist_df, in_dist_data_path, transform=get_transforms('indist', image_size=image_size))
	indist_loader = DataLoader(indist_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

	return indist_loader


def get_outdist_data(batch_size=32, image_size=224):
	base_path = "/root/volume/birds"
	in_dist_data_path = os.path.join(base_path, "u_train_out")
	# get annotations
	outdist_annot_file = "/root/volume/birds/annotation/annotation/anno_u_train_out.json"
	with open(os.path.join(outdist_annot_file), "r", encoding="ISO-8859-1") as file:
		indist = json.load(file)
	outdist_df = pd.DataFrame(indist['images'])
	print(len(outdist_df))
	outdist_dataset = OutDistributionDataset(outdist_df, in_dist_data_path, transform=get_transforms('outdist', image_size=image_size))
	outdist_loader = DataLoader(outdist_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

	return outdist_loader


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
