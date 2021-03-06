#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:37:16 2018

@author: myidispg
"""

import tensorflow as tf 
import numpy as np
from vizdoom import * # Doom environment

import random 
import time # Handling time calculation.
from skimage import transform # Preprocess the image 

from collections import deque # ordered collections with ends
import matplotlib.pyplot as plt 
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

""" Create the environment """

def create_environment():
    game = DoomGame()
    
    # Load the correct configuration
    game.load_config('basic.cfg')
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")
    
    game.init()
    
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions

""" Here we are performing random action to test the environment """

def test_environent():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad") # Map
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode() # Start a new episode
        while not game.is_episode_finished(): # While game is not over
            state = game.get_state() # get current game state
            img = state.screen_buffer() # get image of the game screen
            misc = state.game_variables # Get certain variables from the game like health, ammo etc.
            action = random.choice(actions) # Perform a random action
            print(action)
            reward = game.make_action(action) # get the reward for the action.
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2) # Sleep so that the user can see what is going on.
    game.close()
    
game, possible_actions = create_environment() # Initialize the environment and get possible actions.

"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10,30:-30]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize to 84x84
    preprocessed_frame = transform.resize(normalized_frame, [84,84]) 
    
    return preprocessed_frame

stack_size = 4 # Stack 4 frames

# Inititalize deque of length 4 with zeros as images one array for each image
stacked_frames = deque([np.zeros((84,84), dtype=int) for i in range(stack_size)], max_len = 4)
    
def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocessed frames
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear out the stacked frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
action_size = game.get_available_buttons_size() # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

class DQNetowrk:
    
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_, filters=32, kernel_size=[8,8], strides=[4,4],
                        padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        name="conv1")
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training = True,
                                                                 epsilon = 1e-5,
                                                                 name='batch_norm1')
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4,4],
                                          strides=[2,2], padding='VALID', 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, 
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name='conv2_out') ## --> [9, 9, 64]
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out") ## --> [3,3,128]
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")
            self.output = tf.layers.dense(inputs = self.fc, 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 3, 
                                          activation=None)
            
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)
            
            
        