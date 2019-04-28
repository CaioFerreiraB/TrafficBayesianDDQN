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
from config import Config

#from save_logs import SaveLogs


def calculate_uncertainty(agent, data):
	
	output_list = []

	#Its important to set the network to train mode in order to activate dropout
	agent.policy_net.train()

	#0. Retrieve the outputs from neural network feedfoward n times to build a statistic model
	for i in range(Config.STOCHASTIC_PASSES):
		#print(agent.policy_net(data))
		output_list.append(torch.unsqueeze(F.softmax(agent.policy_net(data)), 0))
		#print(output_list[i])

	agent.policy_net.eval()

	#1. Calculate uncertainty
	uncertainty = torch.cat(output_list, 0).var(0).mean().item()

	#2. Calculate confidence
	output_mean = torch.cat(output_list, 0).mean(0)
	confidence = output_mean.data.cpu().numpy().max()

	return uncertainty, confidence

def check_attack(agent, data):
	#0. Retrieve confidence for the state
	uncertainty, confidence = calculate_uncertainty(agent, data)

	print('uncertainty:', uncertainty)
	print('confidence:', confidence)

	#1. Decide if it's happening or not an attack
	if uncertainty < Config.UNCERTAINTY_TRESSHOLD:
		return True, uncertainty, confidence
	else:
		return False, uncertainty, confidence

def detection_information(attack, detection, SaveLogs):
	if attack: 
		SaveLogs.attacks += 1

	if detection:	
		SaveLogs.attacks_detection += 1
	
	if attack and detection:
		SaveLogs.true_positives += 1
	elif attack and not detection:
		SaveLogs.false_negatives += 1
	elif not attack and detection:
		SaveLogs.false_positives += 1
	elif not attack and not detection:
		SaveLogs.true_negatives += 1


		