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

from model import DQN
from config import Config
from replay_memory import ReplayMemory
from replay_memory import Transition

class Agent:

    def __init__(self, action_set):
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)



    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
            math.exp(-1. * self.steps_done / Config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                a = self.policy_net(state)
                print('Q-values: ', a)
                return a.max(1)[1].view(1, 1)
        else:
            print('random action')
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        print('---------------------------------------------------------------------------------------')
        print('optimizing model')
        if len(self.memory) < Config.BATCH_SIZE:
            return
        print('> sample transitions')
        transitions = self.memory.sample(Config.BATCH_SIZE)
        print('>> ok\n')

        print('> batch transpose')
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))
        print('>> ok\n')

        # Compute a mask of non-final states and concatenate the batch elements
        print('> compute mask of non-final states and concatenate batch elements')
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        print('>> ok\n')

        print('> compute Q(s_t, a) with the policy network')
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        print('>> ok\n')

        print('> Compute V(s_{t+1}) with the target network')
        # Compute V(s_{t+1}) for all next states.
        print('- toch.zeros')
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=self.device)
        print('-- ok')
        print('- target network')
        aux = self.target_net(non_final_next_states)
        print('-- ok')
        print('- aux.max')
        aux = aux.max(1)[0]
        print('-- ok')
        print('- aux.detach')
        aux = aux.detach()
        print('-- ok')
        print('- next_state_values[non_final_mask]')
        next_state_values[non_final_mask] = aux
        print('>> ok\n')

        print('> Compute the expected Q values')
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Config.GAMMA) + reward_batch
        print('>> ok\n')

        print('> Compute Huber loss')
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print('>> ok\n')

        print('> perform backpropagation')
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        print('>> ok\n')


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