import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

LAYERS = {
    'l1': 256,
    'l2': 128,
    'l3': 64,
}


class Critic(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """Critic Network Class

        Args:
            state_size (int): the size of the state space
            action_size (int): the size of the action space
        """
        super(Critic, self).__init__()
        
        self.Q = nn.Sequential(
            nn.Linear(state_size + action_size, LAYERS['l1']),
            nn.ReLU(),
            nn.Linear(LAYERS['l1'], LAYERS['l2']),
            nn.ReLU(),
            nn.Linear(LAYERS['l2'], 1),
        )
    
        
        # initialize the weights using xavier initialization
        self.Q.apply(self._init_weights)
      
    def _init_weights(self, m):
        """Initialize the weights of the network using xavier initialization

        Args:
            m (torch.nn.Module): the network
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    
    def forward(self, state, action):
        """Forward pass for the critic network

        Args:
            state (torch.tensor): the state tensor
            action (torch.tensor): the action tensor

        Returns:
            torch.tensor: the Q value
        """
        x = torch.cat([state, action], dim=-1)
        q = self.Q(x)
        return q


class Actor(torch.nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_low: np.array,
        action_high: np.array,
    ):
        """Actor Network Class

        Args:
            state_size (int): the size of the state space
            action_size (int): the size of the action space
            action_low (np.array): the lower bound of the action space
            action_high (np.array): the upper bound of the action space
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, LAYERS['l1'])
        self.fc2 = nn.Linear(LAYERS['l1'], LAYERS['l2'])
        self.fc_mu = nn.Linear(LAYERS['l2'], action_size)
        self.fc_std = nn.Linear(LAYERS['l2'], action_size)
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
  
        
    def forward(self, state, deterministic=False, with_logprob=True):
        """Forward pass for the actor network

        Args:
            state (torch.tensor): the state tensor

        Returns:
            torch.tensor: the action tensor
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        # We are registering scale and bias as buffers so they can be saved and loaded as part of the model.
        # Buffers won't be passed to the optimizer for training!
        
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        
        if deterministic:
            u = mu
        else:
            u = dist.rsample()
            
        action = torch.tanh(u)
        
        # Enforcing action bounds
        if with_logprob:
            log_prob = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            log_prob = None
        
        return action, log_prob
        