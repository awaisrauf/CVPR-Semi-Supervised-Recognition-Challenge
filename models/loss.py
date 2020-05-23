import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy import CrossEntropyLoss

class LDAMLoss(nn.Module):

	def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
		super(LDAMLoss, self).__init__()
		m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
		m_list = m_list * (max_m / np.max(m_list))
		m_list = torch.cuda.FloatTensor(m_list)
		self.m_list = m_list
		assert s > 0
		self.s = s
		self.weight = weight

	def forward(self, x, target):
		index = torch.zeros_like(x, dtype=torch.uint8)
		index.scatter_(1, target.data.view(-1, 1), 1)

		index_float = index.type(torch.cuda.FloatTensor)
		batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
		batch_m = batch_m.view((-1, 1))
		x_m = x - batch_m

		output = torch.where(index, x_m, x)
		return F.cross_entropy(self.s * output, target, weight=self.weight)


class SoftCrossEntropy(nn.Module):
	def __init__(self):
		super(SoftCrossEntropy, self).__init__()

	def forward(self, predicted, target, model):
		return -(target * torch.log(predicted)).sum(dim=1).mean()


# Cross Entorpy with Regulrizattion
class WL2SP(nn.Module):

	def __init__(self, pretrained_weights, class_weights):
		super(WL2SP, self).__init__()
		self.pretrained_weights = pretrained_weights
		self.class_weights = class_weights

	def forward(self, x, target, model):
		l2_reg_shared_weight = torch.zeros(1).cuda()
		l2_reg = torch.zeros(1)
		for n, m in model.named_modules():
			if "conv" in n:
				try:
					l2_reg_shared_weight += torch.norm(m.weight - self.pretrained_weights[n]) ** 2
				except:
					pass
		# 	print(m)
		# if "fc" in n:
		# 	l2_reg += 0#torch.norm(m.weight)**2

		alpha = 0.1
		beta = 0.01
		loss = F.cross_entropy(x, target,
		                       weight=self.class_weights) + 0.5 * alpha * l2_reg_shared_weight  # + 0.5*beta*l2_reg
		return loss


# Cross Entorpy with Regulrizattion
class UMSE(nn.Module):

	def __init__(self):
		super(UMSE, self).__init__()

	def forward(self, output, model):
		l2_reg_shared_weight = torch.zeros(1).cuda()
		l2_reg = torch.zeros(1)
		for n, m in model.named_modules():
			if "conv" in n:
				try:
					l2_reg_shared_weight += torch.norm(m.weight - self.pretrained_weights[n]) ** 2
				except:
					pass
		# 	print(m)
		# if "fc" in n:
		# 	l2_reg += 0#torch.norm(m.weight)**2

		mse_loss = (output ** 2).mean()
		alpha = 0.1
		beta = 0.01
		loss = mse_loss + 0.5 * alpha * l2_reg_shared_weight  # + 0.5*beta*l2_reg
		return loss
