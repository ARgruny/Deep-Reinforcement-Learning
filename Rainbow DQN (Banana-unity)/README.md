
# Udacity Nanodegree Project

This repository contains a Jupyter Notebook for the Udacity Nanodegree project. Follow the instructions below to set up the environment and run the notebook.

## Prerequisites

- Anaconda or Miniconda (for managing the conda environment)
- Python 3.x

## Setting up the Conda Environment

1. Clone the repository to your local machine:
2. Navigate to the repository directory
3.  Create a new conda environment:
    `conda create --name drlnd python=3.x`
4. Activate the environment:
5. `conda activate drlnd`
6. Install the required packages:
7. `pip install -r requirements.txt`
8. be aware that you need the unityagents package from the Udacity repository: 
   1. git+https://github.com/Udacity/unityagents.git
9.  Create an IPython kernel for the drlnd environment:
10. `python -m ipykernel install --user --name drlnd --display-name "drlnd"`
11. Run Jupyter Notebook
12. `jupyter notebook`
13. Open the notebook `Navigation.ipynb` and change the kernel to `drlnd`
14. Follow the instructions in the notebook to train the agent