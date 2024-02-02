import torch
import torch.nn as nn
import torch.nn.functional as F



class Critic(nn.Module):
    """Critic Network for the PPO Agent"""
    def __init__(self, state_dim: int, hidden_layers: list) -> None:
        """Initialize the Critic Network

        Params
        ======
            state_dim (int): The dimension of the state space
            hidden_layers (list): The hidden layers of the network
        """
        super(Critic, self).__init__()
        self.hidden_layers = hidden_layers

        for i in range(len(hidden_layers)):
            if i == 0:
                self.fc1 = nn.Linear(state_dim, hidden_layers[i])
            else:
                setattr(self, f'fc{i+1}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        self.fc_out = nn.Linear(hidden_layers[-1], 1)

    
    def forward(self, state: torch.tensor) -> torch.tensor:
        """Forward pass of the Critic Network

        Params
        ======
            state (torch.tensor): The state input
        """
        for i in range(len(self.hidden_layers)):
            if i == 0:
                x = F.relu(
                    getattr(self, f'fc{i+1}')(state)    
                )
            else:
                x = F.relu(
                    getattr(self, f'fc{i+1}')(x)
                )

        x = self.fc_out(x)

        return x



class Actor(nn.Module):
    """Actor Network for the PPO Agent"""
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list) -> None:
        """Initialize the Actor Network

        Params
        ======
            state_dim (int): The dimension of the state space
            action_dim (int): The dimension of the action space
            hidden_layers (list): The hidden layers of the network
        """
        super(Actor, self).__init__()
        self.hidden_layers = hidden_layers

        for i in range(len(hidden_layers)):
            if i == 0:
                self.fc1 = nn.Linear(state_dim, hidden_layers[i])
            else:
                setattr(self, f'fc{i+1}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        self.fc_out = nn.Linear(hidden_layers[-1], action_dim)

        self.activation = nn.Tanh()


    def forward(self, state: torch.tensor, deterministic: bool = False) -> torch.tensor:
        """Forward pass of the Actor Network

        Params
        ======
            state (torch.tensor): The state input

        Returns
        =======
            torch.tensor: The action output
        """
        for i in range(len(self.hidden_layers)):
            if i == 0:
                x = self.activation(
                    getattr(self, f'fc{i+1}')(state)    
                )
            else:
                x = self.activation(
                    getattr(self, f'fc{i+1}')(x)
                )

        x = self.fc_out(x)
        prob = F.softmax(x, dim=-1)
        return prob
        