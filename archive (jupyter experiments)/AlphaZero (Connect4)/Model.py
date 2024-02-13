import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *

import numpy as np



class PolicyNetwork6x6(nn.Module):
    """
    Policy Network Class that 
    """
    def __init__(self) -> None:
        super(PolicyNetwork6x6, self).__init__()
        
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


class PolicyNetwork3x3(nn.Module):

    def __init__(self):
        super(PolicyNetwork3x3, self).__init__()
        
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)
        
        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

        
    def forward(self, x):


        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))
        
        
        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.reshape(-1, 9)
        
        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)
        
        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        
        return prob.view(3,3), value