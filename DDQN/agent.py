import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import glob
import os

from model import DQN
from config import Config
from replay_memory import ReplayMemory
from replay_memory import Transition

class Agent:

    def __init__(self, action_set, train=False, load_path=None):
        # Initialize agent params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_set = action_set
        self.action_number = len(action_set)
        self.steps_done = 0
        self.epsilon = Config.EPS_START
        self.episode_durations = []


        # Build networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        if not train: self.policy_net.load(load_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(1000)



    def select_action(self, state, train=True):
        global steps_done
        sample = random.random()
        self.epsilon = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
            math.exp(-1. * self.steps_done / Config.EPS_DECAY)
            
        self.steps_done += 1
        if sample > self.epsilon or (not train):
            with torch.no_grad():
                a = self.policy_net(state)
                return a.max(1)[1].view(1, 1), a.max(0)
        else:
            print('random action')
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long), None

    def optimize_model(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return
        transitions = self.memory.sample(Config.BATCH_SIZE)

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        """
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Config.GAMMA) + reward_batch
        """
        #DDQN TEST
        # Compute argmax Q(s', a; θ)        
        next_state_actions = self.policy_net(non_final_next_states).max(1)[1].detach().unsqueeze(1)

        # Compute Q(s', argmax Q(s', a; θ), θ-)
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Config.GAMMA) + reward_batch


        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def save(self, step, logs_path, label):
        os.makedirs(logs_path + label, exist_ok=True)
        full_label = label + str(step) + '.pth'
        print('full label', full_label)
        logs_path = os.path.join(logs_path, label, full_label)
        self.policy_net.save(logs_path, step=step, optimizer=self.optimizer)
        print('=> Save {}' .format(logs_path)) 
    
    def restore(self, logs_path):
        self.policy_net.load(logs_path)
        self.target_net.load(logs_path)
        print('=> Restore {}' .format(logs_path))
