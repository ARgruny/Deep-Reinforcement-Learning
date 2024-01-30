import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import SACReplayBuffer as ReplayBuffer
from model import Actor, Critic

class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_low: np.array,
        action_high: np.array,
        gamma: float = 0.99,
        lr: float = 0.001,
        batch_size: int = 64,
        tau: float = 0.005,
        max_len: int = int(1e6),
        alpha: float = 0.12,
        update_every: int = 50
    )-> None:
        """SAC Agent

        Args:
            state_size (int): The size of the state space
            action_size (int): The size of the action space
            gamma (float, optional): The discount factor. Defaults to 0.99.
            lr (float, optional): The learning rate. Defaults to 0.0003.
            batch_size (int, optional): The size of the batch. Defaults to 64.
            tau (float, optional): The soft update factor. Defaults to 0.005.
            max_len (int, optional): The size of the replay buffer. Defaults to 100_000.
            target_entropy (float, optional):  The target entropy. Defaults to -1.0.
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.max_len = max_len
        self.action_low = action_low
        self.action_high = action_high
        self.update_every = update_every
        self.alpha = alpha
        
        # step counter
        self.step_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## set up replay buffer
        self.memory = ReplayBuffer(self.max_len)
        
        ## set up critic networks
        self.critic_1 = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_1_target = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_2 = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_2_target = Critic(self.state_size, self.action_size).to(self.device)
        
        # load the weights of the target networks with the local networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # freeze the target networks with the local networks
        for params in self.critic_1_target.parameters():
            params.requires_grad = False
            
        for params in self.critic_2_target.parameters():
            params.requires_grad = False
            
        ## set up actor network
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high).to(self.device)
        # load the weights of the target networks with the local networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        ## set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr)
        
        # log entropy coefficient
        self.target_entropy = torch.tensor(-self.action_size, dtype=float, requires_grad=True, device=self.device)
        self.log_ent_coef = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        

    def act(self, state: np.array, deterministic: bool = False) -> np.array:
        """This method chooses an action based on the current state

        Args:
            state (np.array): The current state

        Returns:
            np.array: The chosen action
        """
        ## choose an action
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, ...]).to(self.device)
            action, _ = self.actor(state, deterministic=deterministic, with_logprob=False)
        
        action = action.cpu().numpy()[0]
        
        return action
    
    
    def action_adapter(self, action: np.array) -> np.array:
        """This method adapts the action to the environment

        Args:
            action (np.array): The chosen action

        Returns:
            np.array: The adapted action
        """
        ## adapt the action to the environment
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    
    def step(self, state: np.array, action: np.array, reward: float, next_state: np.array, done: bool) -> None:
        """This method saves the experience in the replay buffer and updates the networks

        Args:
            state (np.array): The current state
            action (np.array): The chosen action
            reward (float): The reward for the action
            next_state (np.array): The next state
            done (bool): Whether the episode is done or not
        """
        # increase step counter
        self.step_counter += 1
        
        ## save experience in replay buffer
        self.memory.store(state, action, reward, next_state, done)
        
        ## update networks
        if len(self.memory) > self.batch_size:
            if self.step_counter % self.update_every == 0:
                for _ in range(self.update_every):
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences)
            
    
    def update_networks(
        self, states: torch.tensor, 
        actions: torch.tensor, 
        rewards: torch.tensor, 
        next_states: torch.tensor, 
        dones: torch.tensor
    ) -> None:
        """Function to update the critic networks

        Args:
            states (torch.tensor): Torch tensor of states
            actions (torch.tensor): Torch tensor of actions
            rewards (torch.tensor): Torch tensor of rewards
            next_states (torch.tensor): Torch tensor of next states
            dones (torch.tensor): Torch tensor of dones
        """
        
        # ---------------------------- update critic ---------------------------- #
        # calculate the target
        with torch.no_grad():
            next_actions, next_log_probs = self.actor_target(next_states)
            q1_next = self.critic_1_target(next_states, next_actions)
            q2_next = self.critic_2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            target = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs)
        
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        qs = zip([current_q1, current_q2], [self.critic_1_optimizer, self.critic_2_optimizer])
        
        for q, optimizer in qs:
            q_loss = F.mse_loss(q, target)
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        
        for params in self.critic_1.parameters():
            params.requires_grad = False
            
        for params in self.critic_2.parameters():
            params.requires_grad = False
            
        actions, log_probs = self.actor(states)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q_min = torch.min(q1, q2)
        
        a_loss = (self.alpha * log_probs - q_min).mean()
        
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        
        for params in self.critic_1.parameters():
            params.requires_grad = True
            
        for params in self.critic_2.parameters():
            params.requires_grad = True
        
        # ---------------------------- update alpha ---------------------------- #
            
        alpha_loss = -(self.log_ent_coef.exp() * (log_probs.detach() + self.target_entropy).mean())
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()
        self.alpha = self.log_ent_coef.exp()
        
        # ----------------------- update target networks ----------------------- #
        with torch.no_grad():
            self.polyak_update(self.critic_1, self.critic_1_target, self.tau)
            self.polyak_update(self.critic_2, self.critic_2_target, self.tau)
            self.polyak_update(self.actor, self.actor_target, self.tau)
                
    
    def polyak_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        """This method updates the target networks

        Args:
            local_model (nn.Module): The local model
            target_model (nn.Module): The target model
            tau (float): The soft update factor
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    
    def learn(self, experiences: tuple) -> None:
        """This method updates the networks

        Args:
            experiences (tuple): The experiences sampled from the replay buffer
        """
        ## unpack experiences

        states, actions, rewards, next_states, dones = experiences
        
        # ensure the following tensors have the same shape [64, 1]
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        
        ## update critic networks
        self.update_networks(states, actions, rewards, next_states, dones)