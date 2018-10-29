#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:46:48 2018

@author: myidispg
"""

import numpy as np
import gym
import random

env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

qtable = np.zeros((state_size, action_size))

total_episodes = 500000  # Total training episodes
total_test_episodes = 100 # Test epsiode
max_steps = 99 # Max steps per episode

learning_rate = 0.7
gamma = 0.618 # Discount factor

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.01

# List of rewards
rewards = []

# Step 2 for life or until learning is stopped
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        expr_expl_tradeoff = random.uniform(0, 1)
        
        # if this is greater than epsilon, exploitation.
        if expr_expl_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else: # else exploration.
            action = env.action_space.sample()
            
        # Take the action (a) and observe the outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(action)
        
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is current state now
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
        # Reduce epsilon (because as level progresses, exploration must reduce.)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        
        rewards.append(total_rewards)
    print ("Score over time: " +  str(sum(rewards)/total_episodes))
    print(qtable)