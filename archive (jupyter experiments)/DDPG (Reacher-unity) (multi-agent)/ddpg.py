import torch
import torch.nn.functional as F
import numpy as np
from model import Actor, Critic
from buffer import ReplayBuffer
from ounoise import OUNoise
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """The DDPG Agent"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1,
        action_high: float = 1,
        num_agents: int = 1,
        seed: int = 0,
        actor_hidden_units: list = [256, 128],
        critic_hidden_units: list = [256, 256, 128],
        buffer_size: int = int(1e6),
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        weight_decay: float = 0.0,
        grad_clip_max: float = 1.0,
        reward_scale: bool = False,
        add_noise: bool = True,
        update_every: int = 2,
        update_times: int = 20,
    ) -> None:
        """Initialize the DDPG Agent
        
        Params
        ======
            state_dim (int): The dimension of the state space
            action_dim (int): The dimension of the action space
            seed (int): The random seed
            actor_hidden_units (list): The hidden units of the actor network
            critic_hidden_units (list): The hidden units of the critic network
            buffer_size (int): The size of the replay buffer
            batch_size (int): The size of the batch to sample from the replay buffer
            gamma (float): The discount factor
            tau (float): The soft update factor
            lr_actor (float): The learning rate of the actor network
            lr_critic (float): The learning rate of the critic network
            grad_clip_max (float): The maximum value to clip the gradients
            reward_scale (bool): Whether to scale the rewards
            add_noise (bool): Whether to add noise to the actions
            update_every (int): How often to update the networks
            update_times (int): How many times to update the networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.actor_hidden_units = actor_hidden_units
        self.critic_hidden_units = critic_hidden_units
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.grad_clip_max = grad_clip_max
        self.reward_scale = reward_scale
        self.add_noise = add_noise
        self.update_every = update_every
        self.update_times = update_times

        # Actor Network
        self.actor_local = Actor(state_dim, action_dim, seed, actor_hidden_units).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, seed, actor_hidden_units).to(DEVICE)

        # Critic Network
        self.critic_local = Critic(state_dim, action_dim, seed, critic_hidden_units).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim, seed, critic_hidden_units).to(DEVICE)

        ## Hard copy the weights from local to target networks
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(local_param.data)
        

        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Replay Buffer
        self.memory = ReplayBuffer(action_dim, buffer_size, batch_size, seed, DEVICE, reward_scale)

        # Noise Process
        self.noise = OUNoise((self.num_agents, action_dim), seed)

        # Time step
        self.t_step = 0

        # statistics
        self.noise_hist = []


    def step(self, state, action, reward, next_state, done):
        """Save the experience in the replay buffer and learn from it

        Params
        ======
            state (np.array): The current state
            action (np.array): The action taken
            reward (float): The reward received
            next_state (np.array): The next state
            done (bool): Whether the episode is done
        """

        for i in range(self.num_agents):
            a_state = state[i, :]
            a_action = action[i, :]
            a_reward = reward[i]
            a_next_state = next_state[i, :]
            a_done = done[i]

            self.memory.add(a_state, a_action, a_reward, a_next_state, a_done)

        self.t_step = (self.t_step + 1) % self.update_every

        if len(self.memory) > (self.batch_size * self.num_agents):
            for _ in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            if self.t_step == 0:
                self.soft_update(self.critic_local, self.critic_target, self.tau)
                self.soft_update(self.actor_local, self.actor_target, self.tau)

    
    def act(self, state, add_noise=True):
        """Return the action for the given state
        
        Params
        ======
            state (np.array): The current state
            add_noise (bool): Whether to add noise to the action

        Returns
        =======
            np.array: The action to take
        """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        self.actor_local.train()

        if add_noise:
            noise = self.noise.sample()
            action += noise

        actions =  np.clip(action.squeeze(), self.action_low, self.action_high)

        return actions
    

    def reset(self):
        """Reset the noise process"""
        self.noise.reset()


    def learn(self, experiences, gamma):
        """Update the policy and value parameters using given batch of experience tuples

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update the critic
        # Get the predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (1 - dones) * gamma * Q_targets_next

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip_max)
        self.critic_optimizer.step()


        # Update the actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.grad_clip_max)
        self.actor_optimizer.step()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



