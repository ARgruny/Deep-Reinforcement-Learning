# DQN with Double Q-Learning, Dueling Network Architecture and PER (LunarLander-v2)
## Introduction

This is a PyTorch implementation of DQN with Double Q-Learning and Dueling Network Architecture for the LunarLander-v2 environment on OpenAI Gym. The code is based on [this](https://arxiv.org/abs/1511.06581) work and [this](https://arxiv.org/abs/1509.06461) paper.

## Theory

The Dueling Network Architecture is based on the idea that the value of a state is the sum of the value of the state itself and the advantage of each action over the average action. The advantage of an action is the difference between the Q-value of that action and the average Q-value of all actions in that state. The advantage function is defined as:

![Advantage Function](https://latex.codecogs.com/gif.latex?A%28s%2C%20a%29%20%3D%20Q%28s%2C%20a%29%20-%20V%28s%29)

The advantage function is then added to the value function to get the Q-value of each action:

![Q-value](https://latex.codecogs.com/gif.latex?Q%28s%2C%20a%29%20%3D%20V%28s%29%20&plus;%20A%28s%2C%20a%29)

The idea behind Double Q-Learning is that the Q-values are overestimated due to the maximization bias. This is because the same values are used to select and evaluate an action. To overcome this, two separate networks are used to select and evaluate an action. The target network is used to evaluate the action and the main network is used to select the action. The target network is updated with the weights of the main network after a certain number of steps. The target network is then used to evaluate the action for the next number of steps. This is repeated for the entire episode.

## PER Buffer (Prioritized Experience Replay)

The PER buffer is used to store the transitions. The transitions are sampled from the buffer to train the network. The transitions are sampled based on their priority. The priority of a transition is the TD error. The TD error is the difference between the target Q-value and the current Q-value. The TD error is calculated as:

![TD Error](https://latex.codecogs.com/gif.latex?TD%20Error%20%3D%20%7C%20R%20&plus;%20%5Cgamma%20%5Cmax%20Q%28s%27%2C%20a%27%29%20-%20Q%28s%2C%20a%29%20%7C)

The TD error is then added to a small constant to avoid the case where the TD error is zero. The transitions are sampled based on their priority. The transitions with higher priority are sampled more often than the transitions with lower priority. The transitions are updated with the TD error after they are sampled. The transitions with higher priority are updated more often than the transitions with lower priority. The transitions are updated with the TD error to increase the probability of sampling the transitions with higher TD error. The transitions are updated with the TD error as:

![TD Error Update](https://latex.codecogs.com/gif.latex?TD%20Error%20Update%20%3D%20%7C%20R%20&plus;%20%5Cgamma%20%5Cmax%20Q%28s%27%2C%20a%27%29%20-%20Q%28s%2C%20a%29%20%7C%20&plus;%20%5Cepsilon)
