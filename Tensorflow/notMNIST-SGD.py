#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:44:40 2018

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
    del save # help gc to save up memory
    print('Training set- ', train_dataset.shape, train_labels.shape)
    print('Validation set- ', valid_dataset.shape, valid_labels.shape)
    print('Test set- ', test_dataset.shape, test_labels.shape)
    
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # map 0 to [1, 0 , 0 ....]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training even this much data is prohibitive
# Subset the training data for faster turnaround
train_subset = 10000
beta = 0.01

graph = tf.Graph()
with graph.as_default():
    
    #Input the data
    # Load the training, validation and test data into constants that are
    # attached to the graph
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables
    # These are the parameters that we are going to be training. The weight
    # matrix will be initalized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros(num_labels))
    
    # Training computation
    # We multiply the inputs with the weight matrix, and add biases. We compute 
    # its softmax and the cross-entropy(it's one operation in tensorflow, because it is very common
    # it cannot be optimized.) We take the average of this 
    # cross-entropy across all the training examples: that is our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # loss with l2 regularizaation
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss + regularizer * beta)
    
    # Optimizer
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictons for training, validation and test data
    # These are not part of the training, just here to report the accuracy as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
num_steps = 801

def accuracy(predictions, labels):
    return (100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one time operation which ensures the parameters get initialized as 
    # we described in the graph: random_weights for the matrix, zeros for the biases.
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell the .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy array.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if step%100==0:
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy- %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
            # Calling .eval() on valid_prediction is basically like calling run(), but 
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

# Now switch to Stochastic Gradient Descent
# The graph will be similar, except that instead of holding all the training data into a constant node, 
# we create a Placeholder node which will be fed actual data at every call of session.run().
batch_size = 128
beta = 0.01

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For training data, we use a placeholder that will be fed
    # at run time with a training minibatch
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size)) # 128x784
    tf_train_labels = tf.placeholder(tf.float32, shape= [batch_size, num_labels]) # 128x10
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels])) # 784x10
    # truncated_normal gives random values with a normal distribution.
    biases = tf.Variable(tf.zeros([num_labels])) # 10
    
    # Training computation
    logits = tf.matmul(tf_train_dataset, weights) + biases # Intermediate computation step. X*a + b = Y
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits = logits))
    # Cross-entropy sums along the rows. Softmax applies softmax function. Output is a single column.
    # truncated_mean of a single column will return a single value 
    # that is the mean of values along the column.
    
    # loss with l2 regularizaation
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss + regularizer * beta)
    
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # 0.5 is learning rate.
    
    # Predictions for training, test and validation dataset
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
num_steps = 3001

with tf.Session(graph=graph) as session:
    # Initialize all the global variables in the graph.
    tf.global_variables_initializer().run() 
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        # Here only the batch that is generated will be fed to the 
        # computational graph.
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    
    
#  -------Training a neural network with regularization on the dataset ---------------
n_input = image_size * image_size
n_output = num_labels
n_hidden1 = 1024

# Some parameters
learning_rate = 0.4
n_iteration = 1000
batch_size =128
beta = 0.01

graph_nn = tf.Graph()
with graph_nn.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, n_input]) # 128x784
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, n_output]) # 128x10
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    keep_prob = tf.placeholder(tf.float32) # a tensor for the dropout rate which can be updated.
    
    weights = {
            'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1])), # 784x1024
            'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output])) # 64x10
            }
    biases = {
            'b1': tf.Variable(tf.zeros([n_hidden1])), 
            'out': tf.Variable(tf.zeros([n_output]))
            }
    
    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights['w1']) + biases['b1']) # 128x1024 + 1024x1   
    output_layer = tf.matmul(layer_1, weights['out']) + biases['out'] # 128x10 + 10x1
    
    # Original loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits = output_layer))
    
    # L2 loss function regularizaton
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['out'])
    loss = tf.reduce_mean(loss + beta*regularizer) 
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for training, test and validation dataset
    train_prediction = tf.nn.softmax(output_layer)
    
    # for validation dataset predictions
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w1']) + biases['b1'])
    valid_prediction = tf.nn.softmax(tf.matmul(valid_layer_1, weights['out']) + biases['out'])
    
    # for test dataset predictions
    test_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights['w1']) + biases['b1'])
    test_prediction = tf.nn.softmax(tf.matmul(test_layer_1, weights['out']) + biases['out'])
    
num_steps = 3001

valid_scores = {}
with tf.Session(graph=graph_nn) as sess:
    # Initialize all the global variables in the graph.
    tf.global_variables_initializer().run() 
    print("Initialized")
    for step in range(num_steps):
        # Offset for batches
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        # Here only the batch that is generated will be fed to the 
        # computational graph.
        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            
            valid_scores[step] = accuracy(valid_prediction.eval(), valid_labels)
            
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    
# ------------Neural Networks with regularization and dropout--------------------
    
n_input = image_size * image_size
n_output = num_labels
n_hidden1 = 1024

# Some parameters
learning_rate = 0.4
n_iteration = 1000
batch_size =128
dropout = 0.5
beta = 0.01

graph_nn = tf.Graph()
with graph_nn.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, n_input]) # 128x784
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, n_output]) # 128x10
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    weights = {
            'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1])), # 784x1024
            'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output])) # 64x10
            }
    biases = {
            'b1': tf.Variable(tf.zeros([n_hidden1])), 
            'out': tf.Variable(tf.zeros([n_output]))
            }
    
    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights['w1']) + biases['b1']) # 128x1024 + 1024x1 
    # Dropout step 
    keep_prob = tf.placeholder(tf.float32) # a tensor for the dropout rate which can be updated.
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    output_layer = tf.matmul(layer_1, weights['out']) + biases['out'] # 128x10 + 10x1
    
    # Original loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits = output_layer))
    
    # L2 loss function regularizaton
    regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['out'])
    loss = tf.reduce_mean(loss + beta*regularizer) 
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for training, test and validation dataset
    train_prediction = tf.nn.softmax(output_layer)
    
    # for validation dataset predictions
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w1']) + biases['b1'])
    valid_prediction = tf.nn.softmax(tf.matmul(valid_layer_1, weights['out']) + biases['out'])
    
    # for test dataset predictions
    test_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights['w1']) + biases['b1'])
    test_prediction = tf.nn.softmax(tf.matmul(test_layer_1, weights['out']) + biases['out'])
    
num_steps = 3001
valid_scores = {}
with tf.Session(graph=graph_nn) as sess:
    # Initialize all the global variables in the graph.
    tf.global_variables_initializer().run() 
    print("Initialized")
    for step in range(num_steps):
        # Offset for batches
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: dropout}
        # Here only the batch that is generated will be fed to the 
        # computational graph.
        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            
            valid_scores[step] = accuracy(valid_prediction.eval(), valid_labels)
            
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    
    
# Deep Neural Netword with 5 hidden layers, l2 regularization and dropout
n_input = image_size * image_size
n_output = num_labels
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 256
n_hidden4 = 128
n_hidden5 = 64

# Some parameters
learning_rate = 0.4
n_iteration = 1000
batch_size =128
dropout = 0.5
beta = 0.01    

graph_dnn = tf.Graph()
with graph_dnn.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, n_input]) # 128x784
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, n_output]) # 128x10
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    weights = {
            'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1])), # 784x1024
            'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])), # 1024x512
            'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3])), # 512x256
            'w4': tf.Variable(tf.truncated_normal([n_hidden3, n_hidden4])), # 256x128
            'w5': tf.Variable(tf.truncated_normal([n_hidden4, n_hidden5])), # 128x64
            'out': tf.Variable(tf.truncated_normal([n_hidden5, n_output])) # 64x10
            }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden1])), 
            'b2': tf.Variable(tf.zeros([n_hidden2])), 
            'b3': tf.Variable(tf.zeros([n_hidden3])), 
            'b4': tf.Variable(tf.zeros([n_hidden4])), 
            'b5': tf.Variable(tf.zeros([n_hidden5])), 
            'out': tf.Variable(tf.zeros([n_output]))
            }
    """ Training computation """
    
    # Hidden Relu layer 1
    logits_1 = tf.matmul(tf_train_dataset, weights['w1']) + biases['b1']
    logits_1 = tf.nn.relu(logits_1)
     # Dropout
    keep_prob = tf.placeholder(tf.float32)
    logits_1 = tf.nn.dropout(logits_1, keep_prob)
    
    # Hidden Relu layer 2
    logits_2 = tf.matmul(logits_1, weights['w2']) + biases['b2']
    logits_2 = tf.nn.relu(logits_2)
    logits_2 = tf.nn.dropout(logits_2, keep_prob)
    
    # Hidden Relu layer 3
    logits_3 = tf.matmul(logits_2, weights['w3']) + biases['b2']
    logits_3 = tf.nn.relu(logits_3)
    logits_3 = tf.nn.dropout(logits_3, keep_prob)
    
    # Hidden Relu layer 4
    logits_4 = tf.matmul(logits_3, weights['w4']) + biases['b4']
    logits_4 = tf.nn.relu(logits_4)
    logits_4 = tf.nn.dropout(logits_4, keep_prob)
    
    # Hidden Relu layer 5
    logits_5 = tf.matmul(logits_4, weights['w5']) + biases['b5']
    logits_5 = tf.nn.relu(logits_5)
    logits_5 = tf.nn.droput(logits_5, keep_prob)
    
    # Output Layer
    output = tf.matmul(logits_5, weightd['out']) + biases['out']
    
    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output_layer))
    
    regularizers = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + \
                    tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4']) + \
                    tf.nn.l2_loss(weights['w5']) + tf.nn.l2_loss(weights['out'])
                    
    loss = tf.reduce_mean(loss + beta * regularizers)
    
    """ Optimizer """
    global_step = tf. Variable(0)
    start_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staricase = True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    # Predictions for training
    train_prediction = tf.nn.softmax(output)
    
    # Predictions for validation
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w1']) + biases['b1'])
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w2']) + biases['b2'])
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w3']) + biases['b3'])
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w4']) + biases['b4'])
    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['w5']) + biases['b5'])
    valid_prediction = tf.nn.softmax(tf.matmul(valid_layer_5, weights['out']) + biases['out'])
    
    # Predictions for validation
    test_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights['w1']) + biases['b1'])
    test_layer_2 = tf.nn.relu(tf.matmul(test_layer_1, weights['w2']) + biases['b2'])
    test_layer_3 = tf.nn.relu(tf.matmul(test_layer_2, weights['w3']) + biases['b3'])
    test_layer_4 = tf.nn.relu(tf.matmul(test_layer_3, weights['w4']) + biases['b4'])
    test_layer_5 = tf.nn.relu(tf.matmul(test_layer_4, weights['w5']) + biases['b5'])
    test_prediction = tf.nn.softmax(tf.matmul(test_layer_5, weights['out']) + biases['out'])