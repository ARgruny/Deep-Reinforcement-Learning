import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet."""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize weights and bias."""
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))

    def reset_noise(self):
        """Reset noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        """Scale noise for noise distribution."""
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def forward(self, x):
        """Forward method implementation."""
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
        


class NoisyDuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(NoisyDuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        # Dueling DQN Architecture
        # 256, 128, 64
        self.fc1 = NoisyLinear(state_size, fc1_units)
        self.fc2 = NoisyLinear(fc1_units, fc2_units)

        # state value
        self.fc3 = NoisyLinear(fc2_units, fc3_units)
        self.state_value = nn.Linear(fc3_units, 1)

        # action advantage
        self.fc4 = NoisyLinear(fc2_units, fc3_units)
        self.action_advantage = nn.Linear(fc3_units, action_size)

        # reset noise
        self.reset_noise()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        state_value = F.relu(self.fc3(x))
        state_value = self.state_value(state_value)

        action_advantage = F.relu(self.fc4(x))
        action_advantage = self.action_advantage(action_advantage)

        advantage_mean = torch.mean(action_advantage, dim=1, keepdim=True)
        q_value = state_value + action_advantage - advantage_mean

        return q_value
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
    


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        # Dueling DQN Architecture
        # 256, 128, 64
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # state value
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.state_value = nn.Linear(fc3_units, 1)

        # action advantage
        self.fc4 = nn.Linear(fc2_units, fc3_units)
        self.action_advantage = nn.Linear(fc3_units, action_size)

        # initialize weights to xavier
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights to xavier"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.state_value.weight)

        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.action_advantage.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        state_value = F.relu(self.fc3(x))
        state_value = self.state_value(state_value)

        action_advantage = F.relu(self.fc4(x))
        action_advantage = self.action_advantage(action_advantage)

        advantage_mean = torch.mean(action_advantage, dim=1, keepdim=True)
        q_value = state_value + action_advantage - advantage_mean

        return q_value


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # 256, 128, 64
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, action_size)

        # initialize weights to xavier
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights to xavier"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)
