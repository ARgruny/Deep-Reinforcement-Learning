import torch
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cumulative_sum(array, gamma=1.0):
    """Calculate cumulative sum of an array."""
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


def normalize_list(array):
    """Normalize a list."""
    array = np.array(array)
    return (array - array.mean()) / (array.std() + 1e-8)


class Episode:
    """Episode buffer to store experiences."""

    def __init__(self, gamma=0.99, lambd=0.95):
        """Initialize an Episode object.

        Params
        ======
            gamma (float): discount factor
            lambd (float): lambda for GAE
        """
        self.gamma = gamma
        self.lambd = lambd
        self.states = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probs = []

    def append(self, state, action, reward, value, log_prob, scale=20):
        """Add a new experience to memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward/scale)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def end_episode(self, vlaue):
        """Calculate advantages and rewards-to-go."""
        rewards = np.array(self.rewards + [vlaue])
        values = np.array(self.values + [vlaue])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), self.gamma * self.lambd)
        self.rewards_to_go = cumulative_sum(rewards.tolist(), self.gamma)[:-1]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.states)
    

class Memory(Dataset):
    """PPO Replay Memory."""
    def __init__(self) -> None:
        """Initialize a ReplayBuffer object."""
        self.episodes = []
        self.states = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probs = []

    def free_memory(self):
        """Free memory."""
        del self.states[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probs[:]
        del self.episodes[:]

    def append(self, episode):
        self.episodes.append(episode)

    def build(self):
        """Build the memory."""
        for episode in self.episodes:
            self.states.extend(episode.states)
            self.actions.extend(episode.actions)
            self.advantages.extend(episode.advantages)
            self.rewards.extend(episode.rewards)
            self.rewards_to_go.extend(episode.rewards_to_go)
            self.log_probs.extend(episode.log_probs)

        assert (
            len(
                {
                    len(self.states),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probs),
                }
            )
            == 1
        )

        self.advantages = norm



        