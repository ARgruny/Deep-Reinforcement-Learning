# Report for the Tennis environment using Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm

## Introduction
This report contains the results and learnings of training an agent to solve Unity's Tennis environment using the Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm. The code is written in Python and uses PyTorch as the deep learning framework. The implementation is based on the original MADDPG paper and the example implementation of the DDPG algorithm in the Deep Reinforcement Learning Nanodegree from Udacity. This project is part of the Udacity Deep Reinforcement Learning Nanodegree and uses the Unity environment provided by Udacity as well as the libraries and packages provided in the course. The goal of the project is to train an agent to solve the Tennis environment. The environment is considered solved when the average score over 100 episodes is at least +0.5. The report contains a description of the environment, the implementation of the algorithm, the results of training the agent, and possible future improvements. 

## Environment
The environment consists of two agents controlling rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each action is a vector with two numbers, corresponding to movement toward (or away from) the net. The state space has 24 dimensions and the action space has 2 dimensions. The environment is considered solved when the average score over 100 episodes is at least +0.5. To calculate the score for each episode, the maximum score of the two agents is taken.

## Algorithm

### DDPG (Deep Deterministic Policy Gradients)
The Deep Deterministic Policy Gradients (DDPG) algorithm is an off-policy actor-critic algorithm that uses a deterministic policy. It is based on the deterministic policy gradient (DPG) algorithm and combines ideas from the DPG algorithm with ideas from DQN. The algorithm is designed to work with continuous action spaces and uses an actor-critic architecture. The actor is a deterministic policy that learns the best action for a given state. The critic learns the value of the state-action pair and provides feedback to the actor. The algorithm uses a replay buffer to store and sample experiences and a soft update mechanism to update the target networks. The algorithm also uses a noise process to add exploration to the action values. The DDPG algorithm is known to be sample-efficient and stable. The noise used in the DDPG algorithm is usually an Ornstein-Uhlenbeck process, which is a stochastic process that adds noise to the action values and is correlated over time. Other noise processes can also be used.

### MADDPG (Multi-Agent Deep Deterministic Policy Gradients)
The Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm is an extension of the DDPG algorithm to multi-agent environments. It uses the same actor-critic architecture as the DDPG algorithm but trains multiple agents at the same time. The algorithm is designed to work with multi-agent environments and uses a centralized critic and decentralized actor architecture. The centralized critic is a critic that takes the actions and states of all agents as input and learns the value of the global state and the joint action. The decentralized actor is an actor that takes the local state of each agent as input and learns the best action for each agent. The algorithm uses a replay buffer to store and sample experiences and a soft update mechanism to update the target networks. The algorithm also uses a noise process to add exploration to the action values. The MADDPG algorithm is known to be sample-efficient and stable in multi-agent environments.

The MADDPG algorithm is implemented in a way that allows for gathering experiences and then looping through the experiences multiple times to update the networks. This is done to improve the sample efficiency of the algorithm. It also uses polyak averaging to update the target networks. This is done to stabilize the learning process.

The Network architecture used in this implementation is as follows:
- Actor: The actor network consists of an input layer, two hidden layer with 256 and 128 units and a ReLU activation function and an output layer with two units (action space). The forward method calculates the action using the tanh activation function.
- Critic: The critic network consists of an input layer, three hidden layer with 256, 256 and 128 units and a Leaky ReLU activation function and an output layer with one unit. The forward method calculates the Q-value of the state-action pair. In the MADDGP algorithm, the critic takes the actions and states of all agents as input in the second hidden layer. This is coded into the forward pass and the network learns the value of the global state and the joint action.
- The target networks for the actor and the critic are created using the same architecture as the original networks. The target networks are updated using polyak averaging.

The Networks are initialized using a custom weight initialization function. The weights are initialized from a uniform distribution in the range [-3e-3, 3e-3].

The algorithm uses the following hyperparameters:
- Discount factor (gamma): 0.98
- Learning rate (actor): 1e-4
- Learning rate (critic): 3e-4
- Batch size: 256
- Tau (soft update of target networks): 1e-3
- Maximum replay buffer size: 1e5
- Update frequency: 1
- Number of updates: 3
- Target Update Interval: 1
- Noise decay rate: 0.999
- Noise Theta: 0.15
- Noise Sigma: 0.9
- Noise Sigma Min: 0.05

#### The Discount Factor
The discount factor is a value between 0 and 1 that determines the importance of future rewards. A discount factor of 0 means that the agent only cares about the immediate reward, while a discount factor of 1 means that the agent cares about all future rewards equally. A discount factor of 0.98 is used in this implementation.

#### The Learning Rate
The learning rate is a hyperparameter that determines the size of the step the optimizer takes when updating the network weights. A learning rate that is too high can cause the optimizer to overshoot the minimum, while a learning rate that is too low can cause the optimizer to take too long to converge. A learning rate of 1e-4 for the actor and 3e-4 for the critic is used in this implementation. The different learning rates for the actor and the critic are used to stabilize the learning process.

#### The Batch Size
The batch size is the number of experiences sampled from the replay buffer to update the networks. A larger batch size can lead to more stable updates, while a smaller batch size can lead to faster updates. A batch size of 256 is used in this implementation.

#### The Tau
The tau parameter is used to update the target networks. It determines the rate at which the target networks are updated. A smaller tau can lead to more frequent updates, while a larger tau can lead to more stable updates. A tau of 1e-3 is used in this implementation.

#### The Maximum Replay Buffer Size
The maximum replay buffer size is the maximum number of experiences that can be stored in the replay buffer. A larger replay buffer can lead to more stable updates, while a smaller replay buffer can lead to faster updates. A maximum replay buffer size of 1e5 is used in this implementation.

#### The Update Frequency
The update frequency is the number of steps the agent takes before updating the networks. A larger update frequency can lead to more stable updates, while a smaller update frequency can lead to faster updates. An update frequency of 1 is used in this implementation.

#### The Number of Updates
The number of updates is the number of times the networks are updated after gathering experiences. A larger number of updates can lead to more stable updates, while a smaller number of updates can lead to faster updates. A number of updates of 3 is used in this implementation.

#### The Target Update Interval
The target update interval is the number of steps the agent takes before updating the target networks. A larger target update interval can lead to more stable updates, while a smaller target update interval can lead to faster updates. A target update interval of 1 is used in this implementation.

#### The Noise Decay Rate
The noise decay rate is the rate at which the noise process is decayed. A larger noise decay rate can lead to more exploration, while a smaller noise decay rate can lead to more exploitation. A noise decay rate of 0.999 is used in this implementation.

#### The Noise Sigma
The noise sigma parameter is used in the Ornstein-Uhlenbeck process. It determines the volatility of the process. A larger noise sigma can lead to more exploration, while a smaller noise sigma can lead to more exploitation. A noise sigma of 0.9 is used in this implementation.


## Noise Process and Exploration
The noise process used in this implementation is an Normal distribution process instead of the Ornstein-Uhlenbeck process. The noise process is used to add exploration to the action values and is sampled from a normal distribution with a mean of 0 and a standard deviation of 1. The noise is then scaled by a factor that decays over time (Sigma).

## Results
The agent was able to solve the environment in 821 episodes. The average score over 100 episodes was +0.71. The plot below shows the scores of the agent over the episodes and the running average of the scores. I used a higher score threshold of 0.7 to early stop the training to ensure that the agent is able to solve the environment with a good margin.

### Scores for the MADDPG algorithm
![Scores](training_results.png)

## Future Improvements
The implementation of the MADDPG algorithm can be improved in several ways:
- The hyperparameters can be tuned to improve the performance of the agent.
- The network architecture can be modified to improve the learning process.
- The algorithm can be modified to use a single critic for both agents to improve the learning process.
- Parameter sharing can be used to improve the learning process.
- The sample efficiency of the algorithm can be improved by using parallel environments or train for longer.
- Better exploration strategies can be used to improve the learning process. I.e. using a more sophisticated noise process or using parameter space noise. which is a noise process that adds noise to the network weights instead of the action values.
- The algorithm can be modified to use a prioritized experience replay buffer to improve the learning process.


## Learnings
I had difficulties to get the MADDPG algorithm to work. The algorithm was very unstable and I had to try several different hyperparameters. Initially the learning maxed out around an average over 100 episodes of 0.12 which is far from the required 0.5. I found that the noise process had the biggest impact on the learning process. I tried several different noise processes and found that the normal distribution noise process worked best. I also found that the learning rate for the critic had to be higher than the learning rate for the actor to stabilize the learning process further. Generally the algorithm benefits from more samples and longer training. I also found that the algorithm is very sensitive to the hyperparameters and that the learning process can be improved by using a more sophisticated exploration strategy. i.e using a higher noise in the beginning and then decay the noise over time.

## Conclusion
The Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm was able to solve the Tennis environment in 821 episodes. The agent was able to achieve an average score of +0.71 over 100 episodes. The implementation of the algorithm can be improved in several ways to achieve better performance. The algorithm is known for its sample efficiency and stability and is a good choice for solving multi-agent environments.