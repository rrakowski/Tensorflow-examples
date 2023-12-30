#!/usr/bin/env python

# --- Script (using Tensorflow with Keras - high abstraction functions) to build and train a very deep (50 layers) convolutional network model using Residual Networks. 
# The model concerns with learning and detecting numbers (from 0 to 5) from a collection of hand showing sign images

# prepare environment
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
import load_dataset

# construct reusable identity block of layers
def identity_segment(X, f, filters, training=True, initializer=random_uniform):
    """
    arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle Conv's window for the main path
    filters -- python list of integers, defining the number of filters in the Conv layers of the main path
    training -- True: Behave in training mode, False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    returns:
    X -- output of the identity segment, tensor of shape (m, n_H, n_W, n_C)
    """

    # retrieve Filters
    F1, F2, F3 = filters
    
    # save the input value, to add back to the main path. 
    X_shortcut = X
    
    # first component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    
    # Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    return X

# construct reusable iconvolutional block of layers
def convolutional_segment(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    """
    arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle Conv's window for the main path
    filters -- python list of integers, defining the number of filters in the Conv layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode, False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    returns:
    X -- output of the convolutional segment, tensor of shape (n_H, n_W, n_C)
    """

    # retrieve Filters
    F1, F2, F3 = filters
    
    # save the input value
    X_shortcut = X

    # first component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    # third component of main path 
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    
    # shortcut path 
    X = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)

    # add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# build a 50 layer deep Redidual Network (CNN) 
def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    Conv2D -> BATCHNORM -> RELU -> MAXPOOL -> Convsegment -> IDsegment*2 -> Convsegment -> IDsegment*3
    -> Convsegment -> IDsegment*5 -> Convsegment -> IDsegment*2 -> AVGPOOL -> FLATTEN -> DENSE 
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_segment(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_segment(X, 3, [64, 64, 256])
    X = identity_segment(X, 3, [64, 64, 256])
    
    # Stage 3 
    X = convolutional_segment(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_segment(X, 3, [128, 128, 512])
    X = identity_segment(X, 3, [128, 128, 512])
    X = identity_segment(X, 3, [128, 128, 512])

    # Stage 4 
    X = convolutional_segment(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_segment(X, 3, [256, 256, 1024]) 
    X = identity_segment(X, 3, [256, 256, 1024])
    X = identity_segment(X, 3, [256, 256, 1024])
    X = identity_segment(X, 3, [256, 256, 1024])
    X = identity_segment(X, 3, [256, 256, 1024])

    # Stage 5 
    X = convolutional_segment(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_segment(X, 3, [512, 512, 2048]) 
    X = identity_segment(X, 3, [512, 512, 2048]) 

    # Average pooling
    X = AveragePooling2D(2, 2)(X)
   
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # create model
    model = Model(inputs = X_input, outputs = X)
    return model

# build the CNN model
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
print(model.summary())

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

def convert_to_one_hot(Y, classes):
    Y = np.eye(classes)[Y.reshape(-1)].T
    return Y

# convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# train model
model.fit(X_train, Y_train, epochs = 20, batch_size = 32)

# test model's performance
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))