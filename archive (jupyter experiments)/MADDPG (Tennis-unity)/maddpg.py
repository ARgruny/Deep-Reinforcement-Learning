import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import Actor, Critic
from buffer import ReplayBuffer
from ounoise import OUNoise


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """The Multi-Agent DDPG Algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        seed: int = 0,
        hidden_units_actor: list = [256, 256],
        hidden_units_critic: list = [256, 256, 128],
        update_every: int = 1,
        update_times: int = 3,
        batch_size: int = 256,
        buffer_size: int = int(1e5),
        gamma: float = 0.98,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        target_delay: int = 1,
        tau: float = 5e-3,
        reward_scale: bool = False,
    ) -> None:
        """Initialize the MADDPG Agent

        Args:
            state_dim (int): The dimension of the state space
            action_dim (int): The dimension of the action space
            num_agents (int): The number of agents
            seed (int, optional): The random seed. Defaults to 42.
            hidden_units_actor (list, optional): The hidden units of the actor network. Defaults to [256, 256].
            hidden_units_critic (list, optional): The hidden units of the critic network. Defaults to [256, 256, 128].
            update_every (int, optional): How often to update the networks. Defaults to 2.
            update_times (int, optional): How many times to update the networks. Defaults to 4.
            batch_size (int, optional): The size of the batch to sample from the replay buffer. Defaults to 128.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            gamma (float, optional): The discount factor. Defaults to 0.99.
            lr_actor (float, optional): The learning rate of the actor network. Defaults to 1e-3.
            lr_critic (float, optional): The learning rate of the critic network. Defaults to 1e-3.
            tau (float, optional): The soft update factor. Defaults to 1e-3.
            reward_scale (bool, optional): Whether to scale the rewards. Defaults to False.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.seed = torch.manual_seed(seed)
        self.hidden_units_actor = hidden_units_actor
        self.hidden_units_critic = hidden_units_critic
        self.update_every = update_every
        self.update_times = update_times
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.reward_scale = reward_scale
        self.target_delay = target_delay


        critic_input_dim = state_dim * num_agents
        critic_output_dim = action_dim * num_agents

        # Create the actors and critics for each agent
        self.actors = [Actor(state_dim, action_dim, seed, hidden_units_actor).to(DEVICE) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim, seed, hidden_units_actor).to(DEVICE) for _ in range(num_agents)]

        # Create the critics for each agent
        self.critics = [Critic(critic_input_dim, critic_output_dim, seed, hidden_units_critic).to(DEVICE) for _ in range(num_agents)]
        self.target_critics = [Critic(critic_input_dim, critic_output_dim, seed, hidden_units_critic).to(DEVICE) for _ in range(num_agents)]

        # Initialize the target networks with the same weights as the original networks
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())

        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        # Create the optimizers for the actors and critics
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr_actor, weight_decay=0.0) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.lr_critic, weight_decay=0.0) for critic in self.critics]

        # Create the noise process
        self.noise = [OUNoise(action_dim, seed) for _ in range(num_agents)]

        # Create the replay buffer
        self.memory = ReplayBuffer(state_dim, action_dim, buffer_size, batch_size, seed, DEVICE, reward_scale)

        # counter for updating the networks
        self.t_step = 0
        self.t_delay = 0

    def act(self, states: np.array, add_noise: bool = True) -> torch.tensor:
        """Get the actions for each agent

        Args:
            states (np.array): The states for each agent
            add_noise (bool, optional): Whether to add noise to the actions. Defaults to True.

        Returns:
            torch.tensor: The actions for each agent
        """
        actions = []
        for action in range(self.num_agents):

            state = torch.from_numpy(states[action]).float().to(DEVICE)
            self.actors[action].eval()

            with torch.no_grad():
                a = self.actors[action](state).cpu().data.numpy()

            self.actors[action].train()

            if add_noise:
                a += self.noise[action].sample_normal()

            actions.append(a)

        return np.array(actions)
    

    def get_current_sigma(self):
        """Get the current sigma of the noise process

        Returns:
            list: The current sigma for each agent
        """
        return [noise.sigma for noise in self.noise]


    def reset(self, n_eps, curr_eps):
        """Reset the noise process"""
        for noise in self.noise:
            noise.reset(n_eps, curr_eps)


    def save_agents(self, path):
        """Save the actors and critics to a file

        Args:
            path (str): The path to save the file
        """
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), f"{path}/actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{path}/critic_{i}.pth")


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.update_every
        self.t_delay = (self.t_delay + 1) % self.target_delay

        mem_size = len(self.memory)

        if mem_size > self.batch_size and self.t_step == 0:
            for _ in range(self.update_times):
                for i in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, i, self.gamma)
            
            if self.t_delay == 0:
                for i in range(self.num_agents):
                    self.soft_update(self.actors[i], self.target_actors[i], self.tau)
                    self.soft_update(self.critics[i], self.target_critics[i], self.tau)
    

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


    def learn(self, experiences, agent, gamma):
        """For each agent, update the actor and critic networks

        the experiences are the sampled batch from the replay buffer for one agent
        the agent is the index of the agent to update and for which the experiences are sampled
        in MADDPG, the actor and critic networks are updated using the experiences of all agents
        in the experience tuple, the states, actions, rewards, next_states, and dones are stored for all agents
        i.e. states is of shape (batch_size, num_agents, state_size)
        i.e actions is of shape (batch_size, num_agents, action_size)
        i.e rewards is of shape (batch_size, num_agents)
        i.e next_states is of shape (batch_size, num_agents, state_size)
        i.e dones is of shape (batch_size, num_agents)

        the critic is updated using the observations and actions of all agents
        i.e. input to critic is (states, actions) => (batch_size, num_agents * state_size) and (batch_size, num_agents * action_size)
        
        Updating the critic:
        -----------------------
        For each agent, use the sampled experiences to update its critic network. 
        The critic takes as input the observations and actions of all agents (centralized training) 
        and outputs a Q-value representing the expected return. The critic is trained to minimize the mean 
        squared error between its predicted Q-values and the target Q-values, 
        which are computed using the reward and the next state's predicted Q-value from the target critic network 
        (using target actions from the target actor networks of all agents).

        Updating the actor:
        ----------------------
        For each agent, update its actor network by using the policy gradient method. 
        The gradient is calculated with respect to the actions chosen by the actor network, 
        aiming to maximize the expected return as estimated by the critic network. Because the critic is centralized, 
        it can provide a gradient that accounts for the actions of other agents, facilitating coordination or competition.

        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples the states, actions, rewards, next states, and done flags
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        actor = self.actors[agent]
        critic = self.critics[agent]
        target_actor = self.target_actors[agent]
        target_critic = self.target_critics[agent]
        actor_optimizer = self.actor_optimizers[agent]
        critic_optimizer = self.critic_optimizers[agent]

        agent_id = torch.tensor([agent]).to(DEVICE)

        # ---------------------------- update critic ---------------------------- #
        states_full = states.view(-1, self.num_agents * self.state_dim)
        next_states_full = next_states.view(-1, self.num_agents * self.state_dim)
        actions_full = actions.view(-1, self.num_agents * self.action_dim)
        
        # Calculate the target actions for next states from all agents
        with torch.no_grad():
            target_actions = [self.target_actors[agent_id](next_states[:, agent_id, :]) for agent_id in range(self.num_agents)]
            target_actions = torch.cat(target_actions, dim=1)

            # Calculate the target Q-values from the target critic using next states and target actions
            target_q_values = target_critic(next_states_full, target_actions)

            # Compute the target for the current states' Q-values
            target_rewards = rewards[:, agent].view(-1, 1)
            target_dones = dones[:, agent].view(-1, 1)

            q_targets = target_rewards + (gamma * target_q_values * (1 - target_dones))


        # Compute current Q-values using current states and actions
        current_q_values = critic(states_full, actions_full)

        # huber loss
        critic_loss = F.smooth_l1_loss(current_q_values, q_targets.detach())

        # Update the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Detach the actions of other agents to treat them as fixed during actor update
        actions_pred = [actions[:, agent_id, :] if agent_id != agent else actor(states[:, agent_id, :]) for agent_id in range(self.num_agents)]
        actions_pred = torch.cat(actions_pred, dim=1)

        # Calculate the actor loss
        actor_loss = -critic(states_full, actions_pred).mean()

        # Update the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        