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

action = env.action_space.n # The number of columns
states = env.observation_space.n # The number of rows.

# Define some parameters.
