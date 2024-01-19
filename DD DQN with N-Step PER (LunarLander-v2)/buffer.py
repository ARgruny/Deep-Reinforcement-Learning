import torch
import random
import numpy as np
from sumtree import SumTree
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            action_size,
            buffer_size,
            batch_size,
            seed,
            alpha=0.6,
            beta=0.4,
            beta_increment_per_sampling=0.001,
            epsilon=0.01,
            epsilon_increment_per_sampling=0.001):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): prioritization exponent
            beta (float): importance sampling exponent
            beta_increment_per_sampling (float): beta increment per sampling
            epsilon (float): small positive constant to avoid zero priority
            epsilon_increment_per_sampling (float): epsilon increment per sampling
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done",
                "priority"])
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.epsilon_increment_per_sampling = epsilon_increment_per_sampling
        self.seed = random.seed(seed)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, self.max_priority)
        self.memory.append(e)
        self.priorities.append(self.max_priority)

    def max_priority(self):
        """Return maximum priority."""
        return max(self.priorities)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # Calculate priorities
        priorities = np.array(self.priorities)
        priorities = np.power(priorities + self.epsilon, self.alpha)
        priorities /= priorities.sum()

        # Sample experiences
        indices = np.random.choice(
            len(self.memory), self.batch_size, p=priorities)
        experiences = [self.memory[i] for i in indices]

        # Calculate importance sampling weights
        weights = np.power(len(self.memory) * priorities[indices], -self.beta)
        weights /= weights.max()

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Update epsilon
        self.epsilon = min(1.0, self.epsilon +
                           self.epsilon_increment_per_sampling)

        # Convert experiences to tensors
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences."""
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority
        
        # Update max priority
        self.max_priority = max(self.priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __str__(self):
        """Return string representation of internal memory."""
        return str(self.memory)

    def __repr__(self):
        """Return string representation of internal memory."""
        return str(self.memory)

    def __getitem__(self, key):
        """Return item from internal memory."""
        return self.memory[key]

    def __setitem__(self, key, value):
        """Set item in internal memory."""
        self.memory[key] = value



class NstepPrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with N-step returns."""

    def __init__(
            self, 
            buffer_size,
            batch_size,
            alpha,
            beta,
            n_step, 
            gamma
        ):
        """Initialize a PrioritizedReplayBuffer object."""
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.alpha = alpha
        self.beta = beta
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        # Compute n-step return and add to n_step_buffer
        exp = self.experience(state, action, reward, next_state, done)
        self.n_step_buffer.append(exp)

        # When the buffer is filled with n-step returns, add to sum tree
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self._get_n_step_info()
            exp = self.experience(self.n_step_buffer[0].state, self.n_step_buffer[0].action,
                                  reward, next_state, done)
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = 1.0
            self.tree.add(max_p, exp)

    def _get_n_step_info(self):
        """Return the n-step return reward, next state, and done."""
        reward, next_state, done = self.n_step_buffer[-1].reward, self.n_step_buffer[-1].next_state, self.n_step_buffer[-1].done
        for exp in reversed(list(self.n_step_buffer)[:-1]):
            reward = exp.reward + self.gamma * reward * (1 - exp.done)
            if exp.done:
                break
        return reward, next_state, done

    def sample(self):
        """Sample a batch of experiences from the buffer."""
        experiences = []
        indices = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            indices.append(idx)
            experiences.append(data)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(is_weight)).float().to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree.data)