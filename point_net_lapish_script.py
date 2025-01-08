#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:27:00 2024

Script for running the NTPN on the Lapish accumulated value instantaneous versus
delayed lever press task

@author: proxy_loken
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io
import skimage.io as skio

import os
import io

import point_net_utils

import hubdt.b_utils as b_utils
import hubdt.data_loading as data_loading
import hubdt.behav_session_params as behav_session_params
import hubdt.glms as glms


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import power_transform

import point_net
import tensorflow as tf
from tensorflow import keras


# SINGLE SESSION EXPLORATION SCRIPT
# Load single session data for exploration
mat = scipy.io.loadmat('Lapish_data_for_ntpn/label_data.mat')
label_array = mat['label_array']

mat = scipy.io.loadmat('Lapish_data_for_ntpn/spike_data.mat')
spike_array = mat['spike_array']


# reshape arrays for NTPN use
counts = np.unique(label_array[:,0], return_counts=True)

n_neurons = len(counts[0])
n_trials = counts[1][0]

labels = np.reshape(label_array,(n_trials, n_neurons, -1))

spikes = np.reshape(spike_array,(n_trials, n_neurons, -1))

# number of neurons to subsample to
neurons = 32

# select label of interest
# delayed (1) or instantaneous (0)
ys = labels[:,0,2]

# select bins of interest
# first half is LP, second half is outcome
X = spikes[:,:,20:]

X_sub = point_net_utils.subsample_neurons_3d(X, sample_size=neurons, replace=False)
X_sub = np.swapaxes(X_sub, 1, 2)
# Make training and Test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_sub, ys)
# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)


# initialize NTPN
# Create Model (bins, classes, units, neurons)
model = point_net.point_net(20, 2, units=32, dims=neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])

# Train NTPN
# Fit Model
model.fit(train_dataset, epochs=20, validation_data=test_dataset)


# test NTPN





# END OF SINGLE SESSION EXPLORATION SCRIPT



