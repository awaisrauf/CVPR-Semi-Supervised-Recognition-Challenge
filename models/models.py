from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def get_model(args, num_classes, train=False, use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if args.model == "resnet":
		# ResNet
		if args.depth == 18:
			model_ft = models.resnet18(pretrained=use_pretrained)
		elif args.depth == 34:
			model_ft = models.resnet34(pretrained=use_pretrained)
		elif args.depth == 50:
			model_ft = models.resnet50(pretrained=use_pretrained)
		elif args.depth == 101:
			model_ft = models.resnet101(pretrained=use_pretrained)
		elif args.depth == 152:
			model_ft = models.resnet152(pretrained=use_pretrained)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif args.model == "vgg":
		# VGG
		if args.depth == 11:
			model_ft = models.vgg11_bn(pretrained=use_pretrained)
		if args.depth == 13:
			model_ft = models.vgg13_bn(pretrained=use_pretrained)
		elif args.depth == 16:
			model_ft = models.vgg16_bn(pretrained=use_pretrained)
		elif args.depth == 19:
			model_ft = models.vgg19_bn(pretrained=use_pretrained)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif args.model == "efficientnet":
		if args.depth == 0:
			model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, advprop=True)
		elif args.depth == 1:
			model_ft = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes, advprop=True)
		elif args.depth == 2:
			model_ft = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, advprop=True)
		elif args.depth == 3:
			model_ft = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes, advprop=True)
		elif args.depth == 4:
			model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes, advprop=True)
		input_size = 224

	elif args.model == "inceptionv4":
		from pretrainedmodels import inceptionv4
		model_ft = inceptionv4(num_classes=1000, pretrained='imagenet')
		num_ftrs = model_ft.last_linear.in_features[1]
		model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

		input_size = model_ft.input_size

	elif args.model == "inceptionresnetv2":
		from pretrainedmodels import inceptionresnetv2
		model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
		num_ftrs = model_ft.last_linear.in_features
		model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

		input_size = model_ft.input_size[1]

	else:
		print("Invalid model name, exiting...")
		exit()

	# retrained batchnorm

	if args.norm == "rbn" and train:
		for n, m in model_ft.named_modules():
			if "bn" in n:
				m.running_mean.zero_()
				m.running_var.fill_(1)
				m.num_batches_tracked.zero_()

	return model_ft, input_size


if __name__ == "__main__":
	# Initialize the model for this run
	model_ft, input_size = initialize_model(args, 200, train=False, use_pretrained=True)

	# Print the model we just instantiated
	print(model_ft)
