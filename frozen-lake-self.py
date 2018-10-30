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
total_episodes = 150000
learning_rate = 0.8
max_steps = 99
gamma = 0.95 # Discount rate. This is the discount rate for future rewards.

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 1.0

decay_rate = 0.005 # Exponential decay rate for exploration prob.

# List of rewards
rewards = []

#------------TRAINING---------------------------------------------------------------------
# Loop to play 15000 times.
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    # Play each episode a max of 99 times.
    for step in range(max_steps):
        exploration_exploitation_tradeoff = random.uniform(0, 1) # Generate a random number between 0 and 1.
        
        if exploration_exploitation_tradeoff > epsilon:
            action = np.argmax(qtable[state, :]) # Choose the action with most reward for that state.
        else:
            action = env.action_space.sample() # Take any action.
            
        # Take the action decided above and get the new_state, reward, whether game is complete and some auxiallary info.
        new_state, reward, done, info = env.step(action)
        
        # Update the q table accordingly
        # New Q value = Current Q value + 
        #  lr * [Reward + 
        #       discount_rate * (highest Q value between possible actions from the new state s’ )
        #       — Current Q value ]
        
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.argmax(qtable[new_state, :]) - qtable[state,action])
        
        total_rewards += reward
        
        # change the current state to new_state after action is taken.
        state = new_state
        
        # If done, stop this episode
        if done:
            break
        
        # Reduce epsilon to promote exploitation for later stages of the game.
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        
        rewards.append(total_rewards)
    print("Score over time: " + str(sum(rewards)/total_episodes))
    print(qtable)
#--------------------------------------------------------------------------------------------------------    

# Reset the environment because the training is over.
env.reset()
    
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)
    
    for step in range(max_steps):
        
        # Take the action from the trained q table
        action = np.argmax(qtable[state, :])
        print('action for episode ' + str(episode) + ' is ' + str(action))
        
        new_state, reward, done, info = env.step(action)
        
        # if the game is over, render the final state.
        if done:
            env.render()
            
            # Print the number of steps it took.
            print("Number of steps- ", step)
            
            break
        # Update the state accordingly
        state = new_state
        
env.close() # Finally close the environment