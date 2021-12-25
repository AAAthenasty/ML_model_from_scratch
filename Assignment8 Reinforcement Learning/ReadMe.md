# Introduction
This file implement Q-learning with linear function approximation to solve the mountain car environment.\
It implements all of the functions needed to initialize, train, evaluate, and obtain the optimal policies and action values with Q-learning. 

![Alt text](https://github.com/AAAthenasty/ML_model_from_scratch/blob/main/Assignment8%20Reinforcement%20Learning/image.png)
What the Mountain Car environment looks like. The car starts at some point in the valley. The goal is to get to the top right flag.

The state of the environment is represented by two variables, position and velocity.\
Position can be between [−1.2, 0.6] (inclusive) and velocity can be between [−0.07, 0.07] (inclusive). \
These are just measurements along the x-axis. The actions that you may take at any state are {0, 1, 2}, where each number corresponds to an action: (0) pushing the car left, (1) doing nothing, and (2) pushing the car right.


# Q_learning

The Q-learning algorithm is a model-free reinforcement learning algorithm, where we assume we don’t have access to the model of the environment the agent is interacting with. 
We also don’t build a complete model of the environment during the learning process.A learning agent interacts with the environment solely based on calls to step and reset methods of the environment. 
Then the Q-learning algorithm updates the q-values based on the values returned by these methods. Analogously, in the approximation setting the algorithm will instead update the parameters of q-value approximator.
