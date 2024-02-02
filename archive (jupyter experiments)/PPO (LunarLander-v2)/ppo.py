import torch
import copy
import numpy as np
from model import Actor, Critic
from trajectory_buffer import PPOTrajectoryBuffer



class PPOAgent():
    """Proximal Policy Optimization (PPO) agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network: list = [128, 128],
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        lambd: float = 0.95,
        entropy_coef: float = 0.01,
        entropy_coef_decay: float = 0.99,
        clamp_ratio: float = 0.2,
        l2_reg: float = 0.0,
        batch_size: int = 128,
        epochs: int = 10,
        use_adv_norm: bool = False,
    ) -> None:
        """Initialize the PPO Agent

        Params
        ======
            state_dim (int): The dimension of the state space
            action_dim (int): The dimension of the action space
            network (list): The hidden layers of the network
            actor_lr (float): The learning rate of the actor network
            critic_lr (float): The learning rate of the critic network
            gamma (float): The discount factor
            lambd (float): The GAE parameter
            entropy_coef (float): The entropy coefficient
            entropy_coef_decay (float): The entropy coefficient decay
            clamp_ratio (float): The PPO clip ratio
            l2_reg (float): The L2 regularization parameter
            batch_size (int): The size of the mini-batch
            epochs (int): The number of epochs
            use_adv_norm (bool): Whether to use advantage normalization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = network
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.clamp_ratio = clamp_ratio
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_adv_norm = use_adv_norm

        # Initialize the Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, network)
        self.critic = Critic(state_dim, network)

        # Initialize the Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize the Trajectory Buffer
        self.buffer = PPOTrajectoryBuffer(1000, state_dim, action_dim)

        # get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)


    def act(self, state: torch.tensor, deterministic: bool) -> torch.tensor:
        """Select an action from the policy

        Params
        ======
            state (torch.tensor): The current state
            deterministic (bool): Whether to select a deterministic action

        Returns
        =======
            action (torch.tensor): The selected action
            prob_a (float): The probability of the selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.actor(state)
            if deterministic:
                action = torch.argmax(prob)
                prob_a = None
                return action.item(), prob_a
            else:
                m = torch.distributions.Categorical(prob)
                action = m.sample().item()
                prob_a = prob[0][action].item()
                return action, prob_a
            
    
    def step(self, state, action, reward, next_state, done, dw, log_prob, idx):
        """Store the trajectory in the buffer

        Params
        ======
            state (np.array): The current state
            action (np.array): The action taken
            reward (float): The reward received
            next_state (np.array): The next state
            done (bool): Whether the episode is done
            log_prob (float): The log probability of the action
            idx (int): The index to store the trajectory
        """
        self.buffer.states[idx] = torch.tensor(state, dtype=torch.float32)
        self.buffer.actions[idx] = torch.tensor(action, dtype=torch.float32)
        self.buffer.rewards[idx] = torch.tensor(reward, dtype=torch.float32)
        self.buffer.next_states[idx] = torch.tensor(next_state, dtype=torch.float32)
        self.buffer.dones[idx] = torch.tensor(done, dtype=torch.float32)
        self.buffer.dw[idx] = torch.tensor(dw, dtype=torch.float32)
        self.buffer.log_probs[idx] = torch.tensor(log_prob, dtype=torch.float32)
            

    def compute_advantages(
            self, 
            rewards: torch.tensor, 
            dones: torch.tensor, 
            dw: torch.tensor,
            states: torch.tensor, 
            next_states: torch.tensor
        ) -> torch.tensor:
        """Compute the advantages

        Params
        ======
            rewards (torch.tensor): The rewards
            dones (torch.tensor): The done flags
            states (torch.tensor): The states
            next_states (torch.tensor): The next states

        Returns
        =======
            advantages (torch.tensor): The computed advantages
            td_target (torch.tensor): The computed target values
        """
        # Compute the advantages and the target values
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

            deltas = rewards + self.gamma * next_values * (1 - dw) - values
            deltas = deltas.cpu().flatten().numpy()
            advantages = [0]

            for delta, done in zip(deltas[::-1], dones.cpu().flatten().numpy()[::-1]):
                advantage = delta + self.gamma * self.lambd * advantages[-1] * (1 - done)
                advantages.append(advantage)
            
            advantages.reverse()
            advantages = copy.deepcopy(advantages[0:-1])
            advantages = np.array(advantages)
            advantages = torch.tensor(advantages).unsqueeze(1).float().to(self.device)

            td_target = advantages + values

            if self.use_adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        return advantages, td_target
    

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay

        # Get the trajectory from the buffer
        states = self.buffer.states.to(self.device)
        actions = self.buffer.actions.to(self.device)
        rewards = self.buffer.rewards.to(self.device)
        next_states = self.buffer.next_states.to(self.device)
        dones = self.buffer.dones.to(self.device)
        dw = self.buffer.dw.to(self.device)
        old_probs = self.buffer.log_probs.to(self.device)

        # Compute the advantages
        advantages, td_target = self.compute_advantages(rewards, dones, dw, states, next_states)

        # Train the Actor and Critic Networks
        iter_run = int(np.ceil(states.shape[0] / self.batch_size))

        for _ in range(self.epochs):
            # shuffle the data
            permutation = np.arange(states.shape[0])
            np.random.shuffle(permutation)
            permutation = torch.LongTensor(permutation).to(self.device)

            # select mini-batches
            states_perm = states[permutation].clone()
            actions_perm = actions[permutation].clone()
            td_target_perm = td_target[permutation].clone()
            advantages_perm = advantages[permutation].clone()
            old_probs_perm = old_probs[permutation].clone()

            # iterate over mini-batches
            for i in range(iter_run):
                idx = slice(i * self.batch_size, min((i + 1) * self.batch_size, states_perm.shape[0]))

                # actor update
                new_probs = self.actor(states_perm[idx])
                entropy = torch.distributions.Categorical(new_probs).entropy().sum(0, keepdim=True)
                prob_a = new_probs.gather(1, actions_perm[idx])
                ratio = torch.exp(torch.log(prob_a + 1e-10) - torch.log(old_probs_perm[idx] + 1e-10))

                surr1 = ratio * advantages_perm[idx]
                surr2 = torch.clamp(ratio, 1 - self.clamp_ratio, 1 + self.clamp_ratio) * advantages_perm[idx]

                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor_optimizer.step()

                # critic update
                value = self.critic(states_perm[idx])
                tds = td_target_perm[idx]
                critic_loss = (value - tds).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if "weight" in name:
                        critic_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


    def evaluate(self, env, deterministic: bool = True) -> float:
        """Evaluate the agent

        Params
        ======
            env (UnityEnvironment): The Unity environment
            deterministic (bool): Whether to select a deterministic action

        Returns
        =======
            total_reward (float): The total reward
        """
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:

            action, _ = self.act(state, deterministic)
            next_state, reward, done, tr, _ = env.step(action)
            done = done or tr
            total_reward += reward
            state = next_state

        return total_reward