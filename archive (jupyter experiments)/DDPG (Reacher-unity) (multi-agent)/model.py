import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor Network for the DDPG Algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hidden_units: list = [256, 256],
    ) -> None:
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_units = hidden_units
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], action_dim)

        self.reset_params()

    
    def reset_params(self) -> None:
        """Reset the parameters of the network"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state: torch.tensor) -> torch.tensor:
        """Forward pass through the network
        
        Params
        ======
            state (torch.tensor): The input state

        Returns
        =======
            torch.tensor: The output action
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    

class Critic(nn.Module):
    """Critic Network for the DDPG Algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hidden_units: list = [256, 256, 128],
    ) -> None:
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_units = hidden_units

        assert len(hidden_units) == 3, "The critic network must have 3 hidden layers"

        self.first_layer = nn.Linear(state_dim, hidden_units[0])
        self.second_layer = nn.Linear(hidden_units[0] + action_dim, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], 1)

        self.activation = F.leaky_relu

        self.reset_params()


    def reset_params(self) -> None:
        """Reset the parameters of the network"""
        self.first_layer.weight.data.uniform_(*hidden_init(self.first_layer))
        self.second_layer.weight.data.uniform_(*hidden_init(self.second_layer))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        """Forward pass through the network

        Params
        ======
            state (torch.tensor): The input state
            action (torch.tensor): The input action

        Returns
        =======
            torch.tensor: The Q-value
        """
        xs = self.activation(self.first_layer(state))
        x = torch.cat([xs, action], dim=1)
        x = self.activation(self.second_layer(x))
        x = self.activation(self.fc3(x))
        return self.fc4(x)




