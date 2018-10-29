#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:49:05 2018

@author: myidispg
"""


import numpy as np
import gym
import random


env = gym.make("FrozenLake-v0")


action_size = env.action_space.n # number of the columns in the q table
state_size = env.observation_space.n # number of rows in the q table

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 15000 # total episodes
learning_rate = 0.8
max_steps = 99
gamma = 0.95

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0 # exploration probability at start
min_epsilon = 0.1 # minimum exploration probability

decay_rate = 0.005 # Exponential decay rate for exploration prob

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
            
    
env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()



