import torch


class PPOTrajectoryBuffer():
    """Trajectory Buffer for the PPO Agent"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        """Initialize the Trajectory Buffer

        Params
        ======
            capacity (int): The size of the buffer
            state_dim (int): The dimension of the state space
            action_dim (int): The dimension of the action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.int64)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.int64)
        self.dw = torch.zeros((capacity, 1), dtype=torch.int64)
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32)


    def clear(self) -> None:
        """Clear the buffer"""
        self.states = torch.zeros((self.capacity, self.state_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.capacity, self.action_dim), dtype=torch.int64)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((self.capacity, self.state_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.int64)
        self.dw = torch.zeros((self.capacity, 1), dtype=torch.int64)
        self.log_probs = torch.zeros((self.capacity, 1), dtype=torch.float32)
