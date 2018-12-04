import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

# pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# System libraries
import glob
import os
import time

# Project libraries
from model import DQN
from config import Config
from replay_memory import ReplayMemory
from replay_memory import Transition

class Agent:
    """
    The intelligent agent of the simulation. Set the model of the neural network used and general parameters.
    It is responsible to select the actions, optimize the neural network and manage the models.
    """

    def __init__(self, action_set, train=True, load_path=None):
        #1. Initialize agent params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_set = action_set
        self.action_number = len(action_set)
        self.steps_done = 0
        self.epsilon = Config.EPS_START
        self.episode_durations = []

        print('LOAD PATH    --  agent.init:', load_path)
        time.sleep(2)

        #2. Build networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=Config.LEARNING_RATE)

        if not train:
            print('entrou no not train')        
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0)    
            self.policy_net.load(load_path, optimizer=self.optimizer)
            self.policy_net.eval()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(1000)

        


    def select_action(self, state, train=True):
        """
        Selet the best action according to the Q-values outputed from the neural network

        Parameters
        ----------
            state: float ndarray
                The current state on the simulation
            train: bool
                Define if we are evaluating or trainning the model
        Returns
        -------
            a.max(1)[1]: int
                The action with the highest Q-value
            a.max(0): float
                The Q-value of the action taken
        """
        global steps_done
        sample = random.random()
        #1. Perform a epsilon-greedy algorithm
        #a. set the value for epsilon
        self.epsilon = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
            math.exp(-1. * self.steps_done / Config.EPS_DECAY)
            
        self.steps_done += 1

        #b. make the decision for selecting a random action or selecting an action from the neural network
        if sample > self.epsilon or (not train):
            # select an action from the neural network
            with torch.no_grad():
                # a <- argmax Q(s, theta)
                a = self.policy_net(state)
                return a.max(1)[1].view(1, 1), a.max(0)
        else:
            # select a random action
            print('random action')
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long), None

    def optimize_model(self):
        """
        Perform one step of optimization on the neural network
        """

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

    def save(self, step, logs_path, label):
        """
        Save the model on hard disc

        Parameters
        ----------
            step: int
                current step on the simulation
            logs_path: string
                path to where we will store the model
            label: string
                label that will be used to store the model
        """

        os.makedirs(logs_path + label, exist_ok=True)

        full_label = label + str(step) + '.pth'
        logs_path = os.path.join(logs_path, label, full_label)

        self.policy_net.save(logs_path, step=step, optimizer=self.optimizer)
    
    def restore(self, logs_path):
        """
        Load the model from hard disc

        Parameters
        ----------
            logs_path: string
                path to where we will store the model
        """
        self.policy_net.load(logs_path)
        self.target_net.load(logs_path)
