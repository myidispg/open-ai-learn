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

qtable = np.zeros((state_size, action_space))