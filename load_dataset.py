# module to load dataset into py variables

import numpy as np
import h5py

def load_dataset():
    # read in train dataset
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    # train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # your train set labels
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  
    # read in test dataset
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    # test set labels    
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  
    # list of classes
    classes = np.array(test_dataset["list_classes"][:])  
    # reshape label datasets
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
