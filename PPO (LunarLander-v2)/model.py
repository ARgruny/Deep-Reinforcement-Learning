import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERS = {
    'actor': [256, 128, 64],
    'critic': [256, 128, 64]
}

class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, LAYERS['actor'][0])
        self.fc2 = nn.Linear(LAYERS['actor'][0], LAYERS['actor'][1])
        self.fc3 = nn.Linear(LAYERS['actor'][1], LAYERS['actor'][2])
        self.fc4 = nn.Linear(LAYERS['actor'][2], action_size)
        self.l_relu = nn.LeakyReLU(0.1)

    def forward(self, state):
        x = self.l_relu(self.fc1(state))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)


class ValueNetwork(nn.Module):
    
        def __init__(self, state_size):
            super(ValueNetwork, self).__init__()
            self.seed = torch.manual_seed(0)
            self.fc1 = nn.Linear(state_size, LAYERS['critic'][0])
            self.fc2 = nn.Linear(LAYERS['critic'][0], LAYERS['critic'][1])
            self.fc3 = nn.Linear(LAYERS['critic'][1], LAYERS['critic'][2])
            self.fc4 = nn.Linear(LAYERS['critic'][2], 1)
            self.l_relu = nn.LeakyReLU(0.1)
    
        def forward(self, state):
            x = self.l_relu(self.fc1(state))
            x = self.l_relu(self.fc2(x))
            x = self.l_relu(self.fc3(x))
            return self.fc4(x)
    