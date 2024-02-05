import torch
import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed, device, scale_rewards=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): The device to run the model on
            scale_rewards (bool): Whether to scale the rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.scale_rewards = scale_rewards
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        if self.scale_rewards:
            reward = reward / 10.0

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Initialize tensors
        states = torch.zeros((self.batch_size, 2, self.state_size))
        actions = torch.zeros((self.batch_size, 2, self.action_size))
        rewards = torch.zeros((self.batch_size, 2))
        next_states = torch.zeros((self.batch_size, 2, self.state_size))
        dones = torch.zeros((self.batch_size, 2))

        for i, e in enumerate(experiences):
            states[i] = torch.tensor(e.state, dtype=torch.float)
            actions[i] = torch.tensor(e.action, dtype=torch.float)
            rewards[i] = torch.tensor(e.reward, dtype=torch.float)
            next_states[i] = torch.tensor(e.next_state, dtype=torch.float)
            dones[i] = torch.tensor(e.done, dtype=torch.float)
  
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)