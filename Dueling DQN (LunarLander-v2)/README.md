# DQN with Dueling Network Architecture (LunarLander-v2)
## Introduction

This is a PyTorch implementation of DQN with Dueling Network Architecture for the LunarLander-v2 environment on OpenAI Gym. The code is based on [this](https://arxiv.org/abs/1511.06581) work.

## Theory

The Dueling Network Architecture is based on the idea that the value of a state is the sum of the value of the state itself and the advantage of each action over the average action. The advantage of an action is the difference between the Q-value of that action and the average Q-value of all actions in that state. The advantage function is defined as:

![Advantage Function](https://latex.codecogs.com/gif.latex?A%28s%2C%20a%29%20%3D%20Q%28s%2C%20a%29%20-%20V%28s%29)

The advantage function is then added to the value function to get the Q-value of each action:

![Q-value](https://latex.codecogs.com/gif.latex?Q%28s%2C%20a%29%20%3D%20V%28s%29%20&plus;%20A%28s%2C%20a%29)