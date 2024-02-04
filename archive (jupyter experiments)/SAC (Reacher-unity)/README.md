# Solving Unity's Reacher environment using Soft Actor-Critic (SAC) algorithm

## Introduction
This repository contains the code to solve Unity's Reacher environment using the Soft Actor-Critic (SAC) algorithm. The environment is part of the Unity ML-Agents toolkit. The code is written in Python and uses PyTorch as the deep learning framework. The implementation is based on the original SAC paper by Tuomas Haarnoja et al. (2018) and the official SAC implementation by the authors.

## Environment
The environment consists of double-jointed arms that can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. The state space has 33 dimensions and the action space has 4 dimensions.

The environment is considered solved when the average score over 100 episodes is at least +30.

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
    - Version 1: One (1) Agent
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    - Version 2: Twenty (20) Agents
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
7. Place the file in the `unity_env` folder and unzip (or decompress) the file
8. Install the IPython kernel for the created environment with `python -m ipykernel install --user --name drlnd --display-name "drlnd"`
9. Run Jupyter Notebook with `jupyter notebook`
10. Open the notebook `Continious_control.ipynb` and change the kernel to `drlnd`
11. Execute the cells in the notebook to train the agent

### Model weights from training
The model weights can be found in the file `checkpoint_actor.pth`, `checkpoint_critic_1.pth` and `checkpoint_critic_2.pth`

## Running the code
The code is contained in the Jupyter Notebook `Continious_control.ipynb`. The notebook contains the code used for training the agent. It also contains a visualization of the training process and the final results. You simply need to execute the cells in the notebook to train the agent.
