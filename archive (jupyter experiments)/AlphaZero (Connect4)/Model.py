import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *

import numpy as np



class PolicyNetwork(nn.Module):
    """
    Policy Network Class that 
    """
    def __init__(self) -> None:
        super(PolicyNetwork, self).__init__()
        
        # input 6x6 board
        # convert to 5x5x16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        
        # convert to 3x3x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=False)
        
        self.size = 3*3*32
        
        # for actions
        self.fc1_action1 = nn.Linear(self.size, self.size//4)
        self.fc1_action2 = nn.Linear(self.size//4, 36)
        
        # for value
        self.fc1_value1 = nn.Linear(self.size, self.size//6)
        self.fc1_value2 = nn.Linear(self.size//6, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, self.size)
        
        a = F.leaky_relu(self.fc1_action1(x))
        a = self.fc1_action2(a)
        
        avail = (torch.abs(input.squeeze()) != 1).type(torch.FloatTensor)
        avail = avail.reshape(-1, 36)
        maxa = torch.max(a)
        exp = avail * torch.exp(a - maxa)
        prob = exp / torch.sum(exp)
        
        # value head
        v = F.leaky_relu(self.fc1_value1(x))
        v = self.fc1_value2(v)
        v = self.tanh(v)
        
        return prob.view(6, 6), v

