# Solving Unity's Tennis environment using Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm

## Introduction
This repository contains the code to solve Unity's Tennis environment using the Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm. The environment is part of the Unity ML-Agents toolkit. The code is written in Python and uses PyTorch as the deep learning framework. The implementation is based on the original MADDPG paper.

## Environment
The environment consists of two agents controlling rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each action is a vector with two numbers, corresponding to movement toward (or away from) the net. The state space has 24 dimensions and the action space has 2 dimensions.

The environment is considered solved when the average score over 100 episodes is at least +0.5.

To calculate the score for each episode, the maximum score of the two agents is taken.

## Getting Started
### Installation
1. Clone this repository
2. Set up a virtual environment using conda with Python 3.6:
   1. `conda create --name drlnd python=3.6`
3. Activate the environment with `conda activate drlnd`
4. Install the required dependencies using `pip install -r requirements.txt`
5. be aware that you need the unityagents package from the Udacity repository: 
   1. git+https://github.com/Udacity/unityagents.git
6. Download the Unity environment from one of the links below. You need only select the environment that matches your operating system:
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis_Linux.zip)
     - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis_Windows_x86_64.zip)
7. Place the file in the project folder and unzip (or decompress) the file
8. Install the IPython kernel for the created environment with `python -m ipykernel install --user --name drlnd --display-name "drlnd"`
9. Run Jupyter Notebook with `jupyter notebook`
10. Open the notebook `Tennis.ipynb` and change the kernel to `drlnd`
11. Execute the cells in the notebook to train the agent

### Model weights from training
The model weights can be found in the file `checkpoint_actor_0.pth`, `checkpoint_critic_0.pth`, `checkpoint_actor_1.pth` and `checkpoint_critic_1.pth`. These files contain the weights of the trained actor and critic networks for both agents.

## Running the code
The code is contained in the Jupyter Notebook `Tennis.ipynb`. The notebook contains the code used for training the agent. It also contains a visualization of the training process and the final results. You simply need to execute the cells in the notebook to train the agent.
