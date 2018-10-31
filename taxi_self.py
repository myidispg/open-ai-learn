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
total_episodes = 500000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618 # Rate at which future rewards are discounted.

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.01

# A list of rewards
rewards = []

# List of rewards
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    
    for step in range(max_steps):
        
        exploration_exploitation_tradeoff = random.uniform(0, 1)
        
        if exploration_exploitation_tradeoff > epsilon: # Exploit
            action = np.argmax(qtable[state, :]) 
        else:
            action = env.action_space.sample()
            
        new_state, new_reward, done, info = env.step(action)
        
        qtable[state, action] = qtable[state, action] + learning_rate * (new_reward + gamma * 
              np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_reward += new_reward
        
        state = new_state
        
        if done == True:
            break
        
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_reward)
    
    print('Episode ', episode)
    print('Total score over time', str(sum(rewards)))
    
print('Average Score: ', str(sum(rewards)/total_episodes))

print('Final Q-Table- ')
print(qtable)

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    for step in range(max_steps):
        env.render()
        
        action = np.argmax(qtable[state, :])
        
        new_state, reward, done, info = env.step(action)
        
        total_reward += new_reward
        
        if done:
            rewards.append(total_reward)
            # We print the number of step it took.
            print("Score: ", step)
            break
        state = new_state
env.close()


print("Total rewards over time: " + str(sum(rewards)/total_test_episodes))

        
    