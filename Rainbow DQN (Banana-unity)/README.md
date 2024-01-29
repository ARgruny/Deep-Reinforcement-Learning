
# Udacity Nanodegree Project

This repository contains a Jupyter Notebook for the Udacity Nanodegree project. Follow the instructions below to set up the environment and run the notebook.

## Prerequisites

- Anaconda or Miniconda (for managing the conda environment)
- Python 3.x

## Setting up the Conda Environment (Getting Started)

1. Clone the repository to your local machine:
2. Navigate to the repository directory
3.  Create a new conda environment:
    `conda create --name drlnd python=3.x`
4. Activate the environment:
5. `conda activate drlnd`
6. Install the required packages:
7. `pip install -r requirements.txt`
8. Download the unity environment from the Udacity repository:
   1. Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
   2. Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
   3. Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
   4. Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
9. be aware that you need the unityagents package from the Udacity repository: 
   1. git+https://github.com/Udacity/unityagents.git
10. Create an IPython kernel for the drlnd environment:
11. `python -m ipykernel install --user --name drlnd --display-name "drlnd"`
12. Run Jupyter Notebook
13. `jupyter notebook`
14. Open the notebook `Navigation.ipynb` and change the kernel to `drlnd`
15. Follow the instructions in the notebook to train the agent

## model weights from training
the model weights can be found in the file `checkpoint.pth`

## Project Details
The project is based on the Unity ML-Agents environment. The environment is a 3D world with a banana collector agent. The agent can move in four directions and can collect yellow and blue bananas. The goal is to collect as many yellow bananas as possible while avoiding the blue bananas. The environment is considered solved when the agent reaches an average score of 13 over 100 consecutive episodes.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. The action space is discrete and has 4 dimensions. The agent can move forward, backward, turn left and turn right.

The environment is considered solved when the agent reaches an average score of 13 over 100 consecutive episodes.

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent! To train the agent simply run the cells in the notebook. The notebook contains the code used for training. The notebook also contains a visualization of the training process and the final results.

## The following files are included in this repository:

### buffer.py
This file contains the code for the replay buffer used in this agent.

### dqn_agent.py
This file contains the code for the agent used in this project. The agent uses a Deep Q-Network to learn the optimal policy.

### model.py
This file contains the code for the neural network used by the agent.

### sumtree.py
This file contains the code for the sumtree used in the replay buffer.

### Navigation.ipynb
This notebook contains the code used for training the agent. It also contains a visualization of the training process and the final results.