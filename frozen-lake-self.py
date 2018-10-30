#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:10:59 2018

@author: myidispg
"""

import numpy as np
import gym
import random

env = gym.make('FrozenLake-v0')

# Actions will be the columns of the q table
action_size = env.action_space.n
# States are the rows of the q table
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

# Some parameters
total_episode = 15000
learning_rate = 0.8
max_steps = 99
gamma = 0.95

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 1.0

decay_rate = 0.005 # Exponential decay rate for exploration prob.

# List of rewards
rewards = []

# Loop to play 15000 times.
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    # Play each episode a max of 99 times.
    for step in range(max_steps):
        
    
    