[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity DRLND Project 1: Navigation
This is my implementation on Windows 10 of the Navigation Project (Project 1) of Udacity's [Deep Reinforcement Learning Nanodegree (DRLND)](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. Original information about this project can be found in [Udacity's GitHub repository for this project](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

## Introduction

For this project, the agent is to navigate and collect bananas in a large square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 for a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Stated

1. Download the environment from Udacity using this link (for 64-bit Windows only): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip.

2. Place the file in the repository and unzip it there.

## Instructions

Instructions for testing the environment and experimenting with training and testing the agent are provided in `Navigation.ipynb`.

## Dependencies

To set up the Python environment to run the code in the repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   ```bash
   conda create --name ENVNAME python=3.6
   conda activate ENVNAME
   ```

2. Install PyTorch and TorchVision into the newly created environment.

   ```bash
   conda install -c pytorch pytorch torchvision
   ```

3. Navigate to the `python/` folder in the repository and install dependencies specified there.

   ```bash
   pip install .
   ```
