import random
import copy
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.9, sigma_min=0.05, sigma_decay=0.999):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.seed = random.seed(seed)
        self.sigma_decay = sigma_decay
        self.size = size
        self.reset()

    def reset(self, num_episodes=2000, current_episode=0):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        # linear decay of sigma over num_episodes
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)


    def sample(self, sigma=None):
        """Update internal state and return it as a noise sample."""
        if sigma is None:
            sigma = self.sigma
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    

    def sample_normal(self):
        """Return a noise sample."""
        noise = 0.5 * np.random.randn(1, self.size)
        noise = np.clip(noise, -1, 1)
        noise = noise.squeeze()
        # linear decay of sigma over num_episodes
        noise = noise * self.sigma
        return noise