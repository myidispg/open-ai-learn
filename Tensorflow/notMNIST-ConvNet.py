#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:11:35 2018

@author: myidispg
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
# batch_size = 16
# patch_size = 5
# depth = 16
# num_hidden = 64
#
# graph = tf.Graph()
#
# with graph.as_default():
#
#     # Input data
#     # shape- 16x28x28x1
#     tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables
#     # shape- 5x5x1x16 # Filter 1
#     # first and second are filter size, third is number of channels in the input image, last is no.
#     # of convolutional filters. Bias will have the same size as no. of conv filters.
#     layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
#     layer1_biases = tf.Variable(tf.zeros([depth])) # 1x16
#     # shape- 5x5x16x16 # Filter 2
#     layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
#     layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth])) # 1x16
#     # Shape- 784x64 # Filter 3
#     layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#     layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden])) # 1x16
#     # shape- 64x10 # Filter 4
#     layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
#     layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) # 1x10
#
#     # Model.
#     def model(data):
#         # Format = input, kernel, strides, padding. dilation's 1st and last number is always 1.
#         # 1st number in dilation is image number, last is input channel. second and 3rd are strides.
#         conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer1_biases)
#         print(hidden.get_shape().as_list()) # dimensions = numberx14x14x16
#         conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer2_biases)
#         print(hidden.get_shape().as_list())
#         shape = hidden.get_shape().as_list() # dimensions = numberx7x7x16
#         # reshape to dimnesion numberx64 to feed into NN.
#         reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#         # 1st hidden layer in NN
#         hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         print(hidden.get_shape().as_list())
#         # Output layer.
#         return tf.matmul(hidden, layer4_weights) + layer4_biases
#
#     # Training computation.
#     logits = model(tf_train_dataset)
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#     test_prediction = tf.nn.softmax(model(tf_test_dataset))
#
# num_steps = 1001
#
# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print('Initialized')
#   for step in range(num_steps):
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 50 == 0):
#       print('Minibatch loss at step %d: %f' % (step, l))
#       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#
#----CNN with Maxpooling to subsample-------------------
  
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # 16x28x28x1
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, num_channels])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Some variables
    # 1st filter = 5x5x1x16 16 filters so the next layer will have 16 channels.
    # 2nd filter= 5x5x16x16 here, next layer will also have  16 channels
    # 3rd filter = 784 x 64 The output of previous layer will be reshaped.
    # out filter = 64x10. Output layer
    weights = {
            'w1': tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1)),
            'w2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1)),
            'w3': tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)),
            'out': tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
            }
    # 1st bias 1x16
    # 2nd bias 1x 16
    # 3rd bias 1x64
    # 4th bias 1x10
    biases = {
            'b1': tf.Variable(tf.zeros([depth])),
            'b2': tf.Variable(tf.constant(1.0, shape = [depth])),
            'b3': tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            'out': tf.Variable(tf.constant(1.0, shape=[num_labels]))
            }    
    
    def model(data):
        conv = tf.nn.conv2d(data, weights['w1'], strides=[1, 1, 1, 1], padding='SAME') # output = 28x28x16
        hidden = tf.nn.relu(conv + biases['b1'])
        hidden = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output - 14x14x16
        print(hidden.get_shape().as_list())
        conv = tf.nn.conv2d(hidden, weights['w2'], strides = [1, 1, 1, 1], padding='SAME') # output - 14x14x16
        hidden = tf.nn.relu(conv + biases['b2'])
        hidden = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output - 7x7x16
        shape = hidden.get_shape().as_list()
        print(shape) 
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]]) # 1000 x 784 
        hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])  # 1000x784 * 784x64
        return tf.matmul(hidden, weights['out']) + biases['out'] # 1000x64 * 64x10 
    
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    
# ------Net Answer--------------------------
batch_size = 8
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  global_step = tf.Variable(0)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    bias1 = tf.nn.relu(conv1 + layer1_biases)
    pool1 = tf.nn.max_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.nn.relu(conv2 + layer2_biases)
    pool2 = tf.nn.max_pool(bias2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    shape = pool2.get_shape().as_list()
    reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
  # Optimizer.
  learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, }
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
# ----Self convNet with dropout and learning rate decay----------------------

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
beta_regul = 1e-3
drop_out = 0.5

graph = tf.Graph()

with graph.as_default():
    
    # Inout data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)
    
    # Declare some variables
    size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2 # ((28-5+1)//2 - 5 + 1) // 2 = 4
    weights = {
            'w1': tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1)),
            'w2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1)),
            'w3': tf.Variable(tf.truncated_normal([size3 * size3 * depth, num_hidden], stddev = 0.1)), # 256x64
            'w4': tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev = 0.1)),
            'out': tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
            }
    biases = {
            'b1': tf.Variable(tf.zeros([depth])),
            'b2': tf.Variable(tf.constant(1.0, shape=[depth])),
            'b3': tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            'b4': tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            'out': tf.Variable(tf.constant(1.0, shape=[num_labels]))
            }
    
    def model(data, keep_prob):
        # C1 Input 28x28
        conv1 = tf.nn.conv2d(data, weights['w1'], strides=[1,1,1,1], padding='VALID')
        bias1 = tf.add(conv1, biases['b1'])
        # S1 input 24x24
        pool1 = tf.nn.avg_pool(bias1, [1,2,2,1], [1,2,2,1], padding='VALID')
        print(pool1.get_shape().as_list)
        # C2 input 12x12
        conv2 = tf.nn.conv2d(pool1, weights['w2'], strides=[1,1,1,1], padding='VALID')
        bias2 = tf.add(conv2, biases['b2'])
        # S1 input 8x8
        pool2 = tf.nn.avg_pool(bias2, [1,2,2,1], [1,2,2,1], padding='VALID')
        # F3 input 4x4
        shape = pool2.get_shape().as_list()
        print(pool2.get_shape().as_list)
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden1 = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])
        drop1 = tf.nn.dropout(hidden1, keep_prob)
        hidden2 = tf.nn.relu(tf.matmul(drop1, weights['w4']) + biases['b4'])
        drop2 = tf.nn.dropout(hidden2, keep_prob)
        return tf.matmul(drop2, weights['out']) + biases['out']
    
    logits = model(tf_train_dataset, drop_out)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    # Optimizer
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))
    
num_steps = 3001

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:offset+batch_size, :, : :]
        batch_labels = train_labels[offset: offset+batch_size, :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if (step%100==0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
        
        
            
    