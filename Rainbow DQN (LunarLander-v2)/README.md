# DQN

## Using N-Step Prioritized Experience Replay, double Q-Learning and Dueling Network Architecture

This is a PyTorch implementation of DQN with Double Q-Learning and Dueling Network Architecture for the LunarLander-v2 environment on OpenAI Gym.

## Double Q-Learning

In the original DQN paper, the authors used a single network to estimate the Q-values. However, this can lead to overestimation of the Q-values. To overcome this, the authors proposed the use of two networks, one for the selection of the action and the other for the evaluation of the action. This is called Double Q-Learning. The authors showed that this method leads to better performance on Atari games.

## Dueling Network Architecture

The authors of the Dueling Network Architecture paper proposed a new architecture for the DQN. The idea is to separate the network into two streams, one for estimating the state value function and the other for estimating the advantage function. The two streams are then combined to get the Q-values. The authors showed that this method leads to better performance on Atari games.

## N-Step Prioritized Experience Replay

The authors of the N-Step Prioritized Experience Replay paper proposed a new method for sampling from the replay buffer. The idea is to use the N-Step returns for prioritizing the samples in the replay buffer. The authors showed that this method leads to better performance on Atari games.

## Noisy Nets

The authors of the Noisy Nets paper proposed a new method for exploration. The idea is to add noise to the weights of the network. The authors showed that this method leads to better performance on Atari games.

## Prioritized Experience Replay

The authors of the Prioritized Experience Replay paper proposed a new method for sampling from the replay buffer. The idea is to use the TD-Error for prioritizing the samples in the replay buffer. The authors showed that this method leads to better performance on Atari games.

## Sum Tree Data Structure (implementation inspired by [this](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py))

The authors of the Prioritized Experience Replay paper proposed a new data structure for efficient implementation of the Prioritized Experience Replay. The idea is to use a Sum Tree for storing the samples in the replay buffer. The authors showed that this method leads to better performance on Atari games.

## Links to Papers

1. [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
4. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
5. [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf)
