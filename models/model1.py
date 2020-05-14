import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


def get_model(num_classes, train=True, pre_trained='True', norm="bn", train_classifier_only='False'):
	print("==> Creating Model")
	# model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

	if pre_trained=='False':
		model = models.resnet101(pretrained=False)
	else:
		model = models.resnet101(pretrained=True)
	if train_classifier_only == 'True':
		for param in model.parameters():
			param.requires_grad = False
	else:
		pass

	if norm == "rbn" and train == True:
		print("ReTrained BatchNorm")
		for n, m in model.named_modules():
			if "bn" in n:

				m.running_mean.zero_()
				m.running_var.fill_(1)
				m.num_batches_tracked.zero_()
				m.weight.requires_grad = True
				m.bias.requires_grad = True
	#
	model.avgpool = nn.AdaptiveAvgPool2d(1)
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	return model


if __name__ == "__main__":
	print("==> Making the Model")
	m = get_model(200, train=True, pre_trained='True', norm="rbn", train_classifier_only='True')
	print(m)