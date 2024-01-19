# DQN with Double Q-Learning (LunarLander-v2)
## Introduction

This is a PyTorch implementation of DQN with Double Q-Learning for the LunarLander-v2 environment on OpenAI Gym. The code is based on [this](https://arxiv.org/abs/1509.06461) work.

## Theory

The idea behind Double Q-Learning is that the Q-values are overestimated due to the maximization bias. This is because the same values are used to select and evaluate an action. To overcome this, two separate networks are used to select and evaluate an action. The target network is used to evaluate the action and the main network is used to select the action. The target network is updated with the weights of the main network after a certain number of steps. The target network is then used to evaluate the action for the next number of steps. This is repeated for the entire episode.
