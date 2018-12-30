from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import img_as_ubyte, img_as_float
import os, time

def fgsm_attack(data, epsilon, agent, iteration, save_logs):
	# 0. TCreate a copy of the state as a float image. Only used when perturbing the image. To create the perturbations
	#the original data is used.
	data_img = img_as_float(data.detach().cpu().numpy())

	# 1. Set require grad atribute to tensor	
	data = data.double()
	data.requires_grad = True
	
	# 2. Calculate the loss	
	predict = agent.policy_net(data).max(0)[0]
	target = agent.policy_net(data).max(0)[0]
	target_soft = F.softmax(target)
	
	loss = F.smooth_l1_loss(predict, target_soft.detach())
	
	# 3. Calculate gradients of model in backward pass
	loss.backward()

	# 4. Collect datagrad
	data_grad = data.grad.data
	
	# 5. Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()

	# 6. Create the perturbed data by adjusting each pixel of the input data
	perturbed_data = data_img + epsilon*sign_data_grad

	# 7. Adding clipping to maintain [-1,1] range
	perturbed_data = torch.clamp(perturbed_data, -1, 1)
	
	# 8. Remove one dimension to convert into a uint8 image
	perturbed_data = perturbed_data.detach().squeeze().cpu().numpy()
	perturbed_data = perturbed_data.transpose((1, 2, 0))

	# 9. Convert into a uint8 image
	img_int = img_as_ubyte(perturbed_data)

	# 10. Transform the image into a pytorch shape again
	perturbed_data = img_int.transpose((2, 0, 1))
	perturbed_data_int = torch.from_numpy(perturbed_data).to(agent.device).unsqueeze(0)

	#save_logs.save_screenshot(epsilon, iteration, img_int)

	# 11. Return the perturbed data
	return perturbed_data_int

