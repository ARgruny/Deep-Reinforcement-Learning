import numpy as np
import torch
import random
from collections import deque


class SACReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """Replay Buffer for the SAC Agent

        Args:
            capacity (int): The size of the replay buffer
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
        

    def __len__(self):
        return len(self.memory)

    
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer

        Args:
            state (np.array): The current state
            action (np.array): The action taken
            reward (float): The reward received
            next_state (np.array): The next state
            done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
        self.position = (self.position + 1) % self.capacity
        
        
    def sample(self, batch_size: int) -> torch.tensor:
        """Sample a batch from the replay buffer

        Args:
            batch_size (int): The size of the batch to sample

        Returns:
            torch.tensor: The batch of transitions
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )