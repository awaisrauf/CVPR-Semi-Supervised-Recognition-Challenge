#-*- coding:utf-8 -*-
import os
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch


class TrainDataset(Dataset):
	def __init__(self, df, data_path, transform=None):
		self.df = df
		self.labels = df["category_id"]
		self.transform = transform
		self.data_path = data_path

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		file_name = self.df['file_name'].values[idx]
		file_path = os.path.join(self.data_path, file_name)
		image = cv2.imread(file_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# image = Image.open(file_path)
		label = self.labels.values[idx]

		if self.transform:
			try:
				sample = {'image': image, 'label': label}
				augmented = self.transform(**sample)
				image, label = augmented['image'], augmented['label']
			except:
				image = self.transform(image)
		return image, label


class TestDataset(Dataset):
	def __init__(self, df, data_path, transform=None):
		self.df = df
		self.transform = transform
		self.df = df
		self.data_path = data_path

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		file_name = self.df['file_name'].values[idx]
		file_path = os.path.join(self.data_path, file_name)
		image = cv2.imread(file_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		Id = self.df['id'].values[idx]

		if self.transform:
			image = self.transform(image)

		return Id, image


class NoisyStudentDataset(Dataset):
	def __init__(self, df, data_path, transform=None):
		self.df = df
		self.transform = transform
		self.data_path = data_path

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		try:
			data = self.df.iloc[idx].values
			file_name = data[1]
			label = data[2:].astype("float")
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		except:
			idx = idx+2
			data = self.df.iloc[idx].values
			file_name = data[1]
			label = data[2:].astype("float")
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if self.transform:
			try:
				sample = {'image': image, 'label': label}
				augmented = self.transform(**sample)
				image, label = augmented['image'], augmented['label']
			except:
				image = self.transform(image)

		return image, label


class InDistributionDataset(Dataset):
	def __init__(self, df, data_path, transform=None):
		self.df = df
		self.transform = transform
		self.data_path = data_path

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		try:
			file_name = self.df['file_name'].values[idx]
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# image = Image.load(file_path)
			# print(type(image))
			# image = F.to_pil_image(image)
			if self.transform:
				image = self.transform(image)
		except:
			idx=idx+1
			file_name = self.df['file_name'].values[idx]
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# image = Image.load(file_path)
			# print(type(image))
			# image = F.to_pil_image(image)
			if self.transform:
				image = self.transform(image)

		return image, file_name


class OutDistributionDataset(Dataset):
	def __init__(self, df, data_path, transform=None):
		self.df = df
		self.transform = transform
		self.data_path = data_path

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		try:
			file_name = self.df['file_name'].values[idx]
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# image = Image.load(file_path)
			# print(type(image))
			# image = F.to_pil_image(image)
			if self.transform:
				image = self.transform(image)
		except:
			idx=idx+1
			file_name = self.df['file_name'].values[idx]
			file_path = os.path.join(self.data_path, file_name)
			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			if self.transform:
				image = self.transform(image)

		return image

