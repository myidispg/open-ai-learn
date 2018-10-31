#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:40:00 2018

@author: myidispg
"""

import numpy as np
import gym
import random

env = gym.make('Taxi-v2')
env.render()

action_size = env.action_space.n # The number of columns
state_size = env.observation_space.n # The number of rows.

qtable = np.zeros((state_size, action_size))

# Define some parameters.
total_episodes = 5000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618 # Rate at which future rewards are discounted.

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.01

# List of rewards
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        
        exploration_exploitation_tradeoff = random.uniform(0, 1)
        
        if exploration_exploitation_tradeoff > epsilon: # Exploit
            action = np.argmax(qtable[state, :]) 
        else:
            action = env.action_space.sample
            
        new_state, new_reward, done, info = env.step(action)
        
        qtable[state, action] = qtable[state, action] + learning_rate * (new_reward + gamma * np.max(qtable[new_state, :]) + qtable[state, :])
        
        