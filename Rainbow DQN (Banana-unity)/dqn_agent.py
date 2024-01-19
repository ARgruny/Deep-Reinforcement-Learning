import numpy as np
import random

from model import NoisyDuelingQNetwork as DuelingQNetwork
from buffer import NstepPrioritizedReplayBuffer as ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
NSTEP = 3               # number of steps for N-step DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
            state_size, 
            action_size,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            update_every=UPDATE_EVERY,
            n_step=NSTEP,
            seed=random.randint(0, 10000)
        ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_every (int): how often to update the network
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.n_step = n_step
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            alpha=0.6,
            beta=0.4,
            n_step=self.n_step,
            gamma=self.gamma
        )

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        # NoisyNets: no epsilon greedy action selection
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Double Q-learning with N-step targets
        # Get the best action from the local network
        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)

        # Get the Q value of the best action from the target network
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)

        # Compute Q targets for current states (n-step bootstrapping)
        Q_targets = rewards + (gamma ** self.n_step * Q_targets_next.detach() * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute huber loss using PER weights
        loss = (weights * F.smooth_l1_loss(Q_expected, Q_targets, reduction='none')).mean()

        # Update priorities detached from the graph
        priorities = (Q_expected - Q_targets).detach().numpy()
        priorities = np.abs(priorities.squeeze(1))

        # clip priorities to values between 0 and 1000 -> makes LunarLander-v2 more stable
        priorities = np.clip(priorities, 0, 1000)
        self.memory.update_priorities(indices, priorities)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)