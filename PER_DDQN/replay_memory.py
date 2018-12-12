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

"""
This class is used to represent a single transition in the environment
"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    This class is meant to store the transitions in the environment in order to use for trainning the neural network
    """
    def __init__(self, capacity):
        """
        Class constructor. Initializes the capacity of the ReplayMemory 
        args: 
            -capacity: the maximum number of transitions that can be stored
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition. The Transition object requires the following arguments:
            state: the actual state of the envoronment
            action: the action taken on the state
            next_state: the new state reached by taking the action on the previous state
            reward: the reward achieved by reaching the new state 

        If the capacity of the ReplayMemory wasn't reached yet, the Transition will be stored on the end of the array.
        When the capacity be reached, the position will reestart to initial position (0)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Return a randomized sample of the ReplayMemory.
        args:
            batch_size: the size of the sample desired.
        return: 
            a rendomized sample of the ReplayMemory with size 'batch_size'
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)