#!/usr/bin/env python

# --- Script to manually build and train a neural network model using Tensorflow 
# The model concerns with learning and detecting numbers (from 0 to 5) from a collection of hand showing sign images

# prepare environmemt
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

# load dataset
train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

# convert dataset to Tensorflow tensors
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

# normalize and reshap datasets and insert into vectorized tensors
def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, 1)
    and normalize its components.
    
    Arguments
    image - Tensor.
    
    Returns: 
    result -- Transformed tensor 
    """
    image = tf.cast(image, tf.float32) / 256.0
    image = tf.reshape(image, [-1,1])
    return image

new_train = x_train.map(normalize)
new_test = x_test.map(normalize)

# manual version of a function used in forward_propagation func below (unused in the script)
def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)

    Returns: 
    result -- Y = WX + b 
    """
    np.random.seed(1)
    
    X = tf.Variable(np.random.randn(3,1), name = "X")
    W = tf.Variable(np.random.randn(4,3), name = "W")
    b = tf.Variable(np.random.randn(4,1), name = "b")
    Z = tf.add(tf.matmul(W, X), b)
    return Z

# activation function (unused in the script)
def sigmoid(Z):
    """
    Computes the sigmoid of z, arguments:
    z -- input value, scalar or vector

    Returns: 
    result -- (tf.float32) the sigmoid of z
    """
 
    Z = tf.cast("z.", tf.float32)
    A = tf.keras.activations.sigmoid("z")
    return A

# one-hot encoding
def one_hot_matrix(label, depth=6):
    """ 
    Computes the one hot encoding for a single labelArguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take

    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    
    # convert labels to one-hot encoding func
    one_hot = tf.reshape(tf.one_hot(label), (label, depth))
    return one_hot

# map one-hot encodining on dataset labels
new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)

# initialize params (weights and biases)
def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    # initialize parameters                         
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   

    W1 = tf.Variable(initializer(shape=(25, 12288))
    b1 = tf.Variable(initializer(shape=(25, 1))
    W2 = tf.Variable(initializer(shape=(12, 25))
    b2 = tf.Variable(initializer(shape=(12, 1))
    W3 = tf.Variable(initializer(shape=(6, 12))
    b3 = tf.Variable(initializer(shape=(6, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

# call initialized parameters
parameters = initialize_parameters()

# build a forward propagation using Tensorflow with decorator for optimization
@tf.function
def forward_propagation_for_prediction(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # extract parameters dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # calculate NN functions and activations (logits)
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)                                  

    Z2 = tf.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)                                    
 
    Z3 = tf.add(tf.linalg.matmul(W3, A2), b3)
     # YOUR CODE ENDS HERE
    return Z3

# compute cost function
@tf.function
def compute_cost(logits, labels):
    """
    Computes the cost
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = labels, y_pred = logits, from_logits=True))
    return cost

# build 3-layer Tensorflow model and train it using SGD optimizer
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                                      
    
    # initialize parameters
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # use stochastic gradient descent (SGD) optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # divide dataset tensors into minibatches and go extra step to load memory faster
    X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)
    Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) 

    # start training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.
        
        for (minibatch_X, minibatch_Y) in zip(X_train, Y_train):
        
            # Select a minibatch
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(minibatch_X, parameters)
                # 2. compute loss
                minibatch_cost = compute_cost(Z3, minibatch_Y)

            # partially trained weights and biases   
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost / minibatch_size

        # Print the cost for every epoch
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save the parameters in a variable
    print ("Parameters have been trained!")
    return parameters

# call and train the model
model(new_train, new_y_train, new_test, new_y_test, num_epochs=200)

# predict using the above NN model
def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    # create placeholder for input data
    x = tf.placeholder("float", [12288, 1])
    
    # predict labels against input data samples
    z3 = forward_propagation_for_prediction(x, params)

    # retrieve index with max probalility across classes axis
    p = tf.argmax(z3)

    # run Tensorflow session
    sess = tf.Session()

    # get predicted labels for data samples in input dataset
    predictions = sess.run(p, feed_dict={x: X})
    return predictions
