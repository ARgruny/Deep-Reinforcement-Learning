import torch
from model import Policy as PolicyNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class REINFORCE:
    """REINFORCE Agent (Monte Carlo Policy Gradient)"""
    def __init__(
        self,
        action_size,
        state_size,
        gamma=1.0,
        learn_every=1
    ):
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []
        self.eps_counter = 1
        self.learn_every = learn_every
        
        # init network
        self.policy_network = PolicyNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=16
        ).to(device)

        # init optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=1e-2
        )

    def increase_eps_counter(self):
        self.eps_counter += 1

    def reset_memory(self):
        self.rewards = []
        self.log_probs = []
    
    def act(self, state):
        action, log_prom = self.policy_network.act(state)
        return action, log_prom
    
    def step(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def learn(self):

        discounts = [self.gamma**i for i in range(len(self.rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, self.rewards)])

        policy_loss = []
        for log_prob in self.log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        # update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()