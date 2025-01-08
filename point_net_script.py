#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:01:38 2022

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




# OLD SINGLE SESSION VERSION
curr_sess = behav_session_params.load_session_params('i231dlc')
man_behavs = data_loading.load_encodings(curr_sess)
stbin = data_loading.load_stmtx(curr_sess)

# trim stbin to match length, add dimension to match point net architecture(points would be 3-d, this is 1-d)
st_trim = stbin[:len(man_behavs),:]
context_labels, context_array = data_loading.create_context_labels(curr_sess, st_trim)
# network prefers labels from 0-> rather than -1
context_labs_zero = context_labels + 1
# Precut noise version
context_cut, st_cut = b_utils.precut_noise(context_labels, st_trim)
# sample and permute neural data 
st_perm, labels_perm = point_net.gen_permuted_data(st_cut, context_cut, samples=1)
st_pow = power_transform(st_perm)
st_pow = st_pow[:,:,np.newaxis]

X_train, X_val, Y_train, Y_val = train_test_split(st_pow,labels_perm, test_size=0.2)
#create tensorflow style datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
# augment and shuffle
# TODO: add in augmentation
train_dataset = train_dataset.shuffle(len(X_train)).map(point_net.augment).batch(8, drop_remainder=True)
test_dataset = test_dataset.shuffle(len(X_val)).batch(8, drop_remainder=True)









# Single Session version (already loaded, by whatever means) as stbin, context_labels

# Single session selection
selection = [0]

st_cut, context_cut = point_net_utils.remove_noise_cat(stbin, context_labels, selection)

# power transform first, 
X_tsf = point_net_utils.pow_transform(st_cut, selection)

# Alternate std transfrom instead
X_tsf = point_net_utils.std_transform(st_cut, selection)

# Project into 3D with sliding windows
X_sw, Y_sw = point_net_utils.window_projection(X_tsf, context_cut, selection)
X_sw = np.swapaxes(X_sw, 1, 2)

# Subsample neurons
X_sub = point_net_utils.subsample_neurons_3d(X_sw)
# TODO: Add permutation for 3d here

# Make training and Test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_sub, Y_sw)

# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

#
# NEW ACROSS SESSION VERSION
# PROGRAMMATIC LOADING FOR CROSS SESSION DATA
#

# files for dataset and labels

st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'

# load dataset and labels
stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')

# Single session version
#selection = [0]
# pass the single data/label array rather than a list

# Index of sessions to incldue from dataset
selection = [0,1,2,3,4]

# trim noise from st and labels
stcut_list, context_cut_list = point_net_utils.remove_noise_cat(stbin_list, context_list, selection)

    
# Power transform each session
X_tsf = point_net_utils.pow_transform(stcut_list, selection)    

# OR STANDARD TRANSFORM INSTEAD
X_tsf = point_net_utils.std_transform(stcut_list, selection)

# OR RAW
X_tsf = stcut_list

    
# Project into 3D via sliding windows
X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_cut_list, selection)

    
# Within Session Dataset Gen
X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, 32, replace=True)


# Across Session Dataset Gen
X_subs, Ys = point_net_utils.subsample_dataset_3d_across(X_sw_list, Y_sw_list, 2000, 32, True)


# Make training and Test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_subs, Ys)

# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=True)



# COMPILE AND TRAIN
# Point Net takes Batch size, num of classes, num of inputs, input dimension as args
model = point_net.point_net(32, 3, units=32, dims=3)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=20, validation_data=test_dataset)





# Load a second session for comparison
test_sess = behav_session_params.load_session_params('i132')

test_st = data_loading.load_stmtx(curr_sess)

test_st = data_loading.load_stmtx(test_sess)

test_con_labs, test_con_arr = data_loading.create_context_labels(test_sess, test_st)

test_con_labs = test_con_labs + 1
test_std = glms.standardize(test_st)


test_st_perm, test_labs_perm = point_net.gen_permuted_data(test_std, test_con_labs, samples=1)
test_st_perm = test_st_perm[:,:,np.newaxis]

preds = model.predict(test_st_perm)

preds_max = np.argmax(preds, axis=1)

score = accuracy_score(test_con_labs, preds_max)

# combined data

st_perm, labels_perm = point_net.gen_permuted_data(st_trim, context_labs_zero, direction='length',samples=2)

test_st_perm, test_labs_perm = point_net.gen_permuted_data(test_st, test_con_labs, direction='length',samples=1)

comb_st = np.vstack((st_perm, test_st_perm))

comb_labs = np.vstack((labels_perm[:,np.newaxis], test_labs_perm[:,np.newaxis]))

comb_std = glms.standardize(comb_st)

comb_std = comb_std[:,:,np.newaxis]

X_train, X_val, Y_train, Y_val = train_test_split(comb_std,comb_labs, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
# augment and shuffle
# TODO: add in augmentation
train_dataset = train_dataset.shuffle(len(X_train)).map(point_net.augment).batch(32, drop_remainder=True)
test_dataset = test_dataset.shuffle(len(X_val)).batch(32, drop_remainder=True)




# Visualisation Scripts

# Generate Critical Point Sets for an already trained model on a sample of inputs
neut_samples = point_net_utils.select_samples(X_sub, Y_sw, 32, 0)

food_samples = point_net_utils.select_samples(X_sub, Y_sw, 32, 1)

shock_samples = point_net_utils.select_samples(X_sub, Y_sw, 32, 2)

neut_preds = point_net.predict_critical(model, neut_samples, layer_name = 'activation_14')

food_preds = point_net.predict_critical(model, food_samples, layer_name = 'activation_14')

shock_preds = point_net.predict_critical(model, shock_samples, layer_name = 'activation_14')

neut_cs, neut_cs_mean = point_net.generate_critical(neut_preds, 5, neut_samples)

food_cs, food_cs_mean = point_net.generate_critical(food_preds, 5, food_samples)

shock_cs, shock_cs_mean = point_net.generate_critical(shock_preds, 5, shock_samples)

fig = point_net_utils.plot_critical(neut_cs, 5, neut_samples)

fig = point_net_utils.plot_critical(food_cs, 5, food_samples)

fig = point_net_utils.plot_critical(shock_cs, 5, shock_samples)


# Export CS and cs means for cross session analysis

neut_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 0)

food_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 1)

shock_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 2)

neut_preds = point_net.predict_critical(model, neut_samples, layer_name = 'activation_14')

food_preds = point_net.predict_critical(model, food_samples, layer_name = 'activation_14')

shock_preds = point_net.predict_critical(model, shock_samples, layer_name = 'activation_14')

neut_cs, neut_cs_mean = point_net.generate_critical(neut_preds, 1000, neut_samples)

food_cs, food_cs_mean = point_net.generate_critical(food_preds, 1000, food_samples)

shock_cs, shock_cs_mean = point_net.generate_critical(shock_preds, 1000, shock_samples)


neut_cs_dict = {'neut_cs': neut_cs, 'neut_cs_mean': neut_cs_mean}

food_cs_dict = {'food_cs': food_cs, 'food_cs_mean': food_cs_mean}

shock_cs_dict = {'shock_cs': shock_cs, 'shock_cs_mean': shock_cs_mean}

point_net_utils.save_pickle(neut_cs_dict, 'point_net_examples/session_1_neut_cs.p')

point_net_utils.save_pickle(food_cs_dict, 'point_net_examples/session_1_food_cs.p')

point_net_utils.save_pickle(shock_cs_dict, 'point_net_examples/session_1_shock_cs.p')


# Single Session version for trajectory point net

# files for dataset and labels
st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'

# load dataset and labels
stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')

# PARAMS
select = 0
window_size = 32
window_stride = 8
neurons = 11
# PARAMS

# select a single session by index
stbin = stbin_list[select]
context_labels = context_list[select]

# Raw(no transform) workflow with neuron subsampling, no dim reduction
selection = [0]
st_cut, context_cut = point_net_utils.remove_noise_cat(stbin, context_labels, selection)
# Raw transform
X_tsf = st_cut
# Get trajectories with window projection
X_sw, Y_sw = point_net_utils.window_projection(X_tsf, context_cut, selection, window_size=window_size, stride=window_stride)
# swap axes to re-use subsampling function
X_sw = np.swapaxes(X_sw, 1, 2)
# subsample neurons
X_sub = point_net_utils.subsample_neurons_3d(X_sw, sample_size=neurons, replace=False)
X_sub = np.swapaxes(X_sub, 1, 2)
# Make training and Test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_sub, Y_sw)
# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

# Create Model
model = point_net.point_net(32, 3, units=32, dims=neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=20, validation_data=test_dataset)



# Multiple Session Version for Trajectory Point Net

# files for dataset and labels
st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'

# load dataset and labels
stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')

# PARAMS
select = [0,1,3,4]
window_size = 32
window_stride = 8
neurons = 15
# PARAMS

# slice sessions to 'select'
stbin_list = [stbin_list[i] for i in select]
context_list = [context_list[i] for i in select]

# Index of sessions to incldue from dataset
selection = [0,1,2,3]
# trim noise from st and labels
stcut_list, context_cut_list = point_net_utils.remove_noise_cat(stbin_list, context_list, selection)

# RAW transform
X_tsf = stcut_list    
# Project into 3D via sliding windows
X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_cut_list, selection, window_size=window_size, stride=window_stride)
# Within Session Dataset Gen
X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)
# Make training and Test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_subs, Ys)
# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

# Create Model
model = point_net.point_net(32, 3, units=32, dims=neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=20, validation_data=test_dataset)


# addendum for creating fixed train, val, test sets
X_tsf = stcut_list    
# Project into 3D via sliding windows
X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_cut_list, selection, window_size=window_size, stride=window_stride)
# Within Session Dataset Gen
X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)
# Make training and test sets
X_train, X_test, Y_train, Y_test = point_net_utils.train_test_gen(X_subs, Ys)
# make train and val sets from training set
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_train, Y_train)
# save sets for future use
comb_dict = {'X_train':X_train,'X_val':X_val,'X_test':X_test,'Y_train':Y_train,'Y_val':Y_val,'Y_test':Y_test}
point_net_utils.save_pickle(comb_dict,'working_data/combined_data/ablation/context_windowed_ttv_sessions_0134.p')

# separate test session version
X_sw, Y_sw = point_net_utils.window_projection(stcut_list[3], context_cut_list[3], [0], window_size=window_size, stride=window_stride)
X_sw = np.swapaxes(X_sw, 1, 2)
# subsample neurons
X_sub = point_net_utils.subsample_neurons_3d(X_sw, sample_size=neurons, replace=False)
X_sub = np.swapaxes(X_sub, 1, 2)
# save for future use
loo_dict = {'Xs':X_sub,'Ys':Y_sw}

# multiple passes version
stmulti_list = stcut_list + stcut_list + stcut_list
context_multilist = context_cut_list + context_cut_list + context_cut_list
selection = [0,1,2,3,4,5,6,7,8,9,10,11]
X_tsf = stmulti_list    
# Project into 3D via sliding windows
X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_multilist, selection, window_size=window_size, stride=window_stride)
# Within Session Dataset Gen
X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)
# Make training and test sets
X_train, X_test, Y_train, Y_test = point_net_utils.train_test_gen(X_subs, Ys)
# make train and val sets from training set
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_train, Y_train)
# save sets for future use
comb_dict = {'X_train':X_train,'X_val':X_val,'X_test':X_test,'Y_train':Y_train,'Y_val':Y_val,'Y_test':Y_test}
point_net_utils.save_pickle(comb_dict,'working_data/combined_data/ablation/context_multi_windowed_ttv_sessions_0134.p')

# loading from file
ttv_dict = point_net_utils.load_pickle('working_data/combined_data/ablation/context_windowed_ttv_sessions_0134.p')
X_train = ttv_dict['X_train']
X_val = ttv_dict['X_val']
X_test = ttv_dict['X_test']
Y_train = ttv_dict['Y_train']
Y_val = ttv_dict['Y_val']
Y_test = ttv_dict['Y_test']
neurons=15


# load loo version
ttv_dict = point_net_utils.load_pickle('working_data/combined_data/ablation/context_windowed_ttv_sessions_013.p')
X_train = ttv_dict['X_train']
X_val = ttv_dict['X_val']
Y_train = ttv_dict['Y_train']
Y_val = ttv_dict['Y_val']
loo_dict = point_net_utils.load_pickle('working_data/combined_data/ablation/context_windowed_ttv_session_4.p')
X_test = loo_dict['Xs']
Y_test = loo_dict['Ys']
neurons=15

# following load, run this
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)
# Create Model
model = point_net.point_net(32, 3, units=32, dims=neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=40, validation_data=test_dataset)

# running separate test set
predictions = model.predict(X_test)
max_preds = np.argmax(predictions,axis=1)
score = accuracy_score(Y_test, max_preds)

# addendum for running on behaviour 
# files for dataset and labels, manual
st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'
behav_file = 'working_data/combined_data/prop_labels.p'

# hdb version
st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'
behav_file = 'working_data/combined_data/joint_smooth_labels.p'

# load dataset and labels
stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')
stbin_list, behav_list = point_net_utils.load_data_pickle(st_file, behav_file, 'smooth_list')

select = [0,1,3,4]
window_size = 8
window_stride = 2
neurons = 15

stbin_list = [stbin_list[i] for i in select]
context_list = [context_list[i] for i in select]
behav_list = [behav_list[i] for i in select]

selection = [0,1,2,3]

behav_list[0] = behav_list[0][0:25850]
behav_list[1] = behav_list[1][0:26049]
behav_list[2] = behav_list[2][0:26187]
behav_list[3] = behav_list[3][0:26173]

mask0 = context_list[0] != -1
mask1 = context_list[1] != -1
mask2 = context_list[2] != -1
mask3 = context_list[3] != -1

context_cut_list = [0,0,0,0]
behav_cut_list = [0,0,0,0]
stbin_cut_list = [0,0,0,0]

context_cut_list[0] = context_list[0][mask0].copy()
context_cut_list[1] = context_list[1][mask1].copy()
context_cut_list[2] = context_list[2][mask2].copy()
context_cut_list[3] = context_list[3][mask3].copy()

behav_cut_list[0] = behav_list[0][mask0].copy()
behav_cut_list[1] = behav_list[1][mask1].copy()
behav_cut_list[2] = behav_list[2][mask2].copy()
behav_cut_list[3] = behav_list[3][mask3].copy()
stbin_cut_list[0] = stbin_list[0][mask0].copy()
stbin_cut_list[1] = stbin_list[1][mask1].copy()
stbin_cut_list[2] = stbin_list[2][mask2].copy()
stbin_cut_list[3] = stbin_list[3][mask3].copy()

# if using manual prop labels
#behav_cut_list[0][behav_cut_list[0] == 3] = 2
#behav_cut_list[1][behav_cut_list[1] == 3] = 2
#behav_cut_list[2][behav_cut_list[2] == 3] = 2

# if using hdb labels, mask as above to select specific behaviours
mask0 = np.any((behav_cut_list[0] == 31, behav_cut_list[0] == 8, behav_cut_list[0] == 7),axis=0)
mask1 = np.any((behav_cut_list[1] == 31, behav_cut_list[1] == 8, behav_cut_list[1] == 7),axis=0)
mask2 = np.any((behav_cut_list[2] == 31, behav_cut_list[2] == 8, behav_cut_list[2] == 7),axis=0)
mask3 = np.any((behav_cut_list[3] == 31, behav_cut_list[3] == 8, behav_cut_list[3] == 7),axis=0)

context_cut_list[0] = context_cut_list[0][mask0]
context_cut_list[1] = context_cut_list[1][mask1]
context_cut_list[2] = context_cut_list[2][mask2]
context_cut_list[3] = context_cut_list[3][mask3]

behav_cut_list[0] = behav_cut_list[0][mask0]
behav_cut_list[1] = behav_cut_list[1][mask1]
behav_cut_list[2] = behav_cut_list[2][mask2]
behav_cut_list[3] = behav_cut_list[3][mask3]
stbin_cut_list[0] = stbin_cut_list[0][mask0]
stbin_cut_list[1] = stbin_cut_list[1][mask1]
stbin_cut_list[2] = stbin_cut_list[2][mask2]
stbin_cut_list[3] = stbin_cut_list[3][mask3]

for i in range(len(behav_cut_list)):
    behav_cut_list[i][behav_cut_list[i]==31] = 0
    behav_cut_list[i][behav_cut_list[i]==8] = 1
    behav_cut_list[i][behav_cut_list[i]==7] = 2

X_tsf = stbin_cut_list
comb_labels = [np.vstack((behav_cut_list[i],context_cut_list[i])).T for i in range(len(behav_cut_list))]

X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_cut_list, selection, window_size=window_size, stride=window_stride)
X_sw_list, Y_bh_list = point_net_utils.window_projection(X_tsf, behav_cut_list, selection, window_size=window_size, stride=window_stride)

X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)
X_subs, Ybhs = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_bh_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)

# store context labels for plotting later
Y_cons = Ys
# this is the labelset you are using to train/test
Ys = Ybhs
# force balanced class split, as beheviours are unequally distributed
X_train, X_val, Y_train, Y_val = point_net_utils.split_balanced(X_subs, Ys)



# Alternate version for segmentation

# hdb version
st_file = 'working_data/combined_data/raw_stbins.p'
context_file = 'working_data/combined_data/context_labels.p'
behav_file = 'working_data/combined_data/joint_smooth_labels.p'

# load dataset and labels
stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')
stbin_list, behav_list = point_net_utils.load_data_pickle(st_file, behav_file, 'smooth_list')

select = [0,1,3,4]
window_size = 64
window_stride = 32
neurons = 15

stbin_list = [stbin_list[i] for i in select]
context_list = [context_list[i] for i in select]
behav_list = [behav_list[i] for i in select]

selection = [0,1,2,3]

behav_list[0] = behav_list[0][0:25850]
behav_list[1] = behav_list[1][0:26049]
behav_list[2] = behav_list[2][0:26187]
behav_list[3] = behav_list[3][0:26173]

mask0 = context_list[0] != -1
mask1 = context_list[1] != -1
mask2 = context_list[2] != -1
mask3 = context_list[3] != -1

context_cut_list = [0,0,0,0]
behav_cut_list = [0,0,0,0]
stbin_cut_list = [0,0,0,0]

context_cut_list[0] = context_list[0][mask0].copy()
context_cut_list[1] = context_list[1][mask1].copy()
context_cut_list[2] = context_list[2][mask2].copy()
context_cut_list[3] = context_list[3][mask3].copy()

behav_cut_list[0] = behav_list[0][mask0].copy()
behav_cut_list[1] = behav_list[1][mask1].copy()
behav_cut_list[2] = behav_list[2][mask2].copy()
behav_cut_list[3] = behav_list[3][mask3].copy()
stbin_cut_list[0] = stbin_list[0][mask0].copy()
stbin_cut_list[1] = stbin_list[1][mask1].copy()
stbin_cut_list[2] = stbin_list[2][mask2].copy()
stbin_cut_list[3] = stbin_list[3][mask3].copy()


# Optional: Remove noise category

mask0 = behav_cut_list[0] != -1
mask1 = behav_cut_list[1] != -1
mask2 = behav_cut_list[2] != -1
mask3 = behav_cut_list[3] != -1

behav_cut_list[0] = behav_cut_list[0][mask0]
behav_cut_list[1] = behav_cut_list[1][mask1]
behav_cut_list[2] = behav_cut_list[2][mask2]
behav_cut_list[3] = behav_cut_list[3][mask3]
stbin_cut_list[0] = stbin_cut_list[0][mask0]
stbin_cut_list[1] = stbin_cut_list[1][mask1]
stbin_cut_list[2] = stbin_cut_list[2][mask2]
stbin_cut_list[3] = stbin_cut_list[3][mask3]
context_cut_list[0] = context_cut_list[0][mask0]
context_cut_list[1] = context_cut_list[1][mask1]
context_cut_list[2] = context_cut_list[2][mask2]
context_cut_list[3] = context_cut_list[3][mask3]




X_tsf = stbin_cut_list
comb_labels = [np.vstack((behav_cut_list[i],context_cut_list[i])).T for i in range(len(behav_cut_list))]

# each window needs a context label (as normal) and behaviour labels for each point in the window
# neurons, context labels
X_sw_list, Y_sw_list = point_net_utils.window_projection(X_tsf, context_cut_list, selection, window_size=window_size, stride=window_stride)
# behav labels, context labels
Y_bh_list, Y_cons_list = point_net_utils.window_projection_segments(behav_cut_list, context_cut_list, selection, window_size=window_size, stride=window_stride)
# subsample neurons
X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, neurons, replace=False)
X_subs = np.swapaxes(X_subs,1,2)
# combine behav labels into one vector
Ybhs = np.concatenate(Y_bh_list, axis=0)

# For NTPN function, all labels need to be >=0. 'noise' will be 0 after this adjustment
#Ybhs = Ybhs + 1

# generate training and test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_subs, Ybhs)
# generate tensors
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

# Build segmentation model
num_classes = np.max(Ybhs)+1
model = point_net.point_net_segment(64, num_classes, units=32, dims=neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=40, validation_data=test_dataset)



# VISUALISATION SCRIPTS FOR RAW SAMPLES AND CRITICAL SETS

# behavioural version for generating critical sets, uppers, etc.
# necessitated by the need to preserve multiple label sets
neut_samples, neut_inds = point_net_utils.select_samples(X_subs, Ys, 100, 0,return_index=True)
food_samples, food_inds = point_net_utils.select_samples(X_subs, Ys, 100, 1,return_index=True)
shock_samples, shock_inds = point_net_utils.select_samples(X_subs, Ys, 100, 2,return_index=True)
neut_preds = point_net.predict_critical(model, neut_samples, layer_name = 'activation_14')
food_preds = point_net.predict_critical(model, food_samples, layer_name = 'activation_14')
shock_preds = point_net.predict_critical(model, shock_samples, layer_name = 'activation_14')
# NOTE: in order to preserve the indices of the labels, generate_critical has to be called with matching num_samples, and len(samples)
neut_cs, neut_cs_mean = point_net.generate_critical(neut_preds, 100, neut_samples)
food_cs, food_cs_mean = point_net.generate_critical(food_preds, 100, food_samples)
shock_cs, shock_cs_mean = point_net.generate_critical(shock_preds, 100, shock_samples)
# extract matched labels for each set
neut_blabels = Ybhs[neut_inds]
neut_clabels = Y_cons[neut_inds]
food_blabels = Ybhs[food_inds]
food_clabels = Y_cons[food_inds]
shock_blabels = Ybhs[shock_inds]
shock_clabels = Y_cons[shock_inds]

neut_cs_dict = {'neut_cs': neut_cs, 'neut_cs_mean': neut_cs_mean, 'neut_samples': neut_samples, 'neut_blabels':neut_blabels, 'neut_clabels':neut_clabels}
food_cs_dict = {'food_cs': food_cs, 'food_cs_mean': food_cs_mean, 'food_samples': food_samples, 'food_blabels':food_blabels, 'food_clabels':food_clabels}
shock_cs_dict = {'shock_cs': shock_cs, 'shock_cs_mean': shock_cs_mean, 'shock_samples': shock_samples, 'shock_blabels':shock_blabels, 'shock_clabels':shock_clabels}

point_net_utils.save_pickle(neut_cs_dict, 'point_net_examples/trajectories/session_0134_b31n_cs.p')
point_net_utils.save_pickle(food_cs_dict, 'point_net_examples/trajectories/session_0134_b08f_cs.p')
point_net_utils.save_pickle(shock_cs_dict, 'point_net_examples/trajectories/session_0134_b07s_cs.p')


# generate critical sets
neut_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 0)
food_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 1)
shock_samples = point_net_utils.select_samples(X_sub, Y_sw, 1000, 2)
neut_preds = point_net.predict_critical(model, neut_samples, layer_name = 'activation_14')
food_preds = point_net.predict_critical(model, food_samples, layer_name = 'activation_14')
shock_preds = point_net.predict_critical(model, shock_samples, layer_name = 'activation_14')
neut_cs, neut_cs_mean = point_net.generate_critical(neut_preds, 1000, neut_samples)
food_cs, food_cs_mean = point_net.generate_critical(food_preds, 1000, food_samples)
shock_cs, shock_cs_mean = point_net.generate_critical(shock_preds, 1000, shock_samples)

# save CS and samples to pickle
neut_cs_dict = {'neut_cs': neut_cs, 'neut_cs_mean': neut_cs_mean, 'neut_samples': neut_samples}
food_cs_dict = {'food_cs': food_cs, 'food_cs_mean': food_cs_mean, 'food_samples': food_samples}
shock_cs_dict = {'shock_cs': shock_cs, 'shock_cs_mean': shock_cs_mean, 'shock_samples': shock_samples}

point_net_utils.save_pickle(neut_cs_dict, 'point_net_examples/trajectories/session_013_neut_cs.p')
point_net_utils.save_pickle(food_cs_dict, 'point_net_examples/trajectories/session_013_food_cs.p')
point_net_utils.save_pickle(shock_cs_dict, 'point_net_examples/trajectories/session_013_shock_cs.p')



# project samples and CS down to 3-D for trajectory plotting
num_samples = 5

# PCA Version
neut_pca_cs, neut_pca_samps = point_net_utils.pca_cs_windowed(neut_cs, neut_samples, dims=3)
food_pca_cs, food_pca_samps = point_net_utils.pca_cs_windowed(food_cs, food_samples, dims=3)
shock_pca_cs, shock_pca_samps = point_net_utils.pca_cs_windowed(shock_cs, shock_samples, dims=3)

# select a random subset to plot
neut_pca_css, neut_pca_sampss = point_net_utils.select_samples_cs(neut_pca_cs, neut_pca_samps, num_samples)
food_pca_css, food_pca_sampss = point_net_utils.select_samples_cs(food_pca_cs, food_pca_samps, num_samples)
shock_pca_css, shock_pca_sampss = point_net_utils.select_samples_cs(shock_pca_cs, shock_pca_samps, num_samples)

fig = point_net_utils.plot_critical(neut_pca_css, 5, neut_pca_sampss)
fig = point_net_utils.plot_critical(food_pca_css, 5, food_pca_sampss)
fig = point_net_utils.plot_critical(shock_pca_css, 5, shock_pca_sampss)

# UMAP Version
num_samples = 50
# subsample before UMAP to save computation
neut_css, neut_sampss = point_net_utils.select_samples_cs(neut_cs, neut_samples, num_samples)
food_css, food_sampss = point_net_utils.select_samples_cs(food_cs, food_samples, num_samples)
shock_css, shock_sampss = point_net_utils.select_samples_cs(shock_cs, shock_samples, num_samples)
# run UMAP
neut_umap_cs, neut_umap_samps = point_net_utils.umap_cs_windowed(neut_css, neut_sampss, dims=3)
food_umap_cs, food_umap_samps = point_net_utils.umap_cs_windowed(food_css, food_sampss, dims=3)
shock_umap_cs, shock_umap_samps = point_net_utils.umap_cs_windowed(shock_css, shock_sampss, dims=3)
# subselect again if need be
num_samples = 5
neut_umap_css, neut_umap_sampss = point_net_utils.select_samples_cs(neut_umap_cs, neut_umap_samps, num_samples)
food_umap_css, food_umap_sampss = point_net_utils.select_samples_cs(food_umap_cs, food_umap_samps, num_samples)
shock_umap_css, shock_umap_sampss = point_net_utils.select_samples_cs(shock_umap_cs, shock_umap_samps, num_samples)
# plot
fig = point_net_utils.plot_critical(neut_umap_css, 5, neut_umap_sampss)
fig = point_net_utils.plot_critical(food_umap_css, 5, food_umap_sampss)
fig = point_net_utils.plot_critical(shock_umap_css, 5, shock_umap_sampss)


# OPTIONAL SECTION FOR SEPARATE TEST SESSION FOR COMPARISON
test_cut, test_context = point_net_utils.remove_noise_cat(stbin_list[3], context_list[3], [0])
test_X_sw, test_Y_sw = point_net_utils.window_projection(test_cut, test_context, [0], window_size=window_size, stride=window_stride)
test_X_sw = np.swapaxes(test_X_sw, 1, 2)
test_X_sub = point_net_utils.subsample_neurons_3d(test_X_sw, sample_size=neurons, replace=False)
test_X_sub = np.swapaxes(test_X_sub, 1, 2)

test_neut_samples = point_net_utils.select_samples(test_X_sub, test_Y_sw, 500, 0)
test_food_samples = point_net_utils.select_samples(test_X_sub, test_Y_sw, 500, 1)
test_shock_samples = point_net_utils.select_samples(test_X_sub, test_Y_sw, 500, 2)

test_neut_preds = point_net.predict_critical(model, test_neut_samples, layer_name = 'activation_14')
test_food_preds = point_net.predict_critical(model, test_food_samples, layer_name = 'activation_14')
test_shock_preds = point_net.predict_critical(model, test_shock_samples, layer_name = 'activation_14')

test_neut_cs, test_neut_cs_mean = point_net.generate_critical(test_neut_preds, 500, test_neut_samples)
test_food_cs, test_food_cs_mean = point_net.generate_critical(test_food_preds, 500, test_food_samples)
test_shock_cs, test_shock_cs_mean = point_net.generate_critical(test_shock_preds, 500, test_shock_samples)

test_neut_cs_dict = {'neut_cs': test_neut_cs, 'neut_cs_mean': test_neut_cs_mean, 'neut_samples': test_neut_samples}
test_food_cs_dict = {'food_cs': test_food_cs, 'food_cs_mean': test_food_cs_mean, 'food_samples': test_food_samples}
test_shock_cs_dict = {'shock_cs': test_shock_cs, 'shock_cs_mean': test_shock_cs_mean, 'shock_samples': test_shock_samples}

point_net_utils.save_pickle(test_neut_cs_dict, 'point_net_examples/trajectories/session_test4_neut_cs.p')
point_net_utils.save_pickle(test_food_cs_dict, 'point_net_examples/trajectories/session_test4_food_cs.p')
point_net_utils.save_pickle(test_shock_cs_dict, 'point_net_examples/trajectories/session_test4_shock_cs.p')

tneut_pca_cs, tneut_pca_samps = point_net_utils.pca_cs_windowed(test_neut_cs, test_neut_samples, dims=3)
tfood_pca_cs, tfood_pca_samps = point_net_utils.pca_cs_windowed(test_food_cs, test_food_samples, dims=3)
tshock_pca_cs, tshock_pca_samps = point_net_utils.pca_cs_windowed(test_shock_cs, test_shock_samples, dims=3)

# END OF OPTIONAL TEST SECTION

# preparing CS for alternate plotting
# load CS if necessary
neut_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_neut_cs.p')
food_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_food_cs.p')
shock_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_shock_cs.p')

food_all_cs = food_cs_dict['food_cs']
neut_all_cs = neut_cs_dict['neut_cs']
shock_all_cs = shock_cs_dict['shock_cs']

# extract unique points from each trajectory
food_uniques = []
neut_uniques = []
shock_uniques = []
min_size_f = food_cs_dict['food_cs_mean']
min_size_n = neut_cs_dict['neut_cs_mean']
min_size_s = shock_cs_dict['shock_cs_mean']
# TODO: the min sizes of some of the trajectories is too small, find another way to set this and threshold

food_uniques, min_size_f = point_net_utils.cs_extract_uniques(food_all_cs, food_cs_dict['food_cs_mean'])
neut_uniques, min_size_n = point_net_utils.cs_extract_uniques(neut_all_cs, neut_cs_dict['neut_cs_mean'])
shock_uniques, min_size_s = point_net_utils.cs_extract_uniques(shock_all_cs, shock_cs_dict['shock_cs_mean'])

food_sub_cs = point_net_utils.cs_subsample(food_uniques, min_size_f)
neut_sub_cs = point_net_utils.cs_subsample(neut_uniques, min_size_n)
shock_sub_cs = point_net_utils.cs_subsample(shock_uniques, min_size_s)

# Generate a PCA/UMAP projection of the whole set of samples(trajectories), 
all_sub_cs = np.concatenate((neut_sub_cs, food_sub_cs, shock_sub_cs))
Y_sub_cs = np.zeros(all_sub_cs.shape[0])
Y_sub_cs[len(neut_sub_cs):len(neut_sub_cs)+len(food_sub_cs)] = 1
Y_sub_cs[-len(shock_sub_cs):] = 2

# flatten for direct UMAP
X_flat_cs = np.reshape(all_sub_cs,(all_sub_cs.shape[0],-1))
u_model = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0)

X_flat_cs_umap = u_model.fit_transform(X_flat_cs)

# then use those weightings to project
# the subsamples CS points into 3D, then plot those individually

# TODO: for CCA, choose a good looking trajectory from among the CS samples. This is our 'Y' target. Use CCA to align
# other trajectories with this one


# Lorenz Atrtactor Trajectory Point Net

from openml.datasets.functions import get_dataset

from gtda.time_series import SlidingWindow

# pull lorenz dataset from openml database
# will be deprecated in future versions, modify accordingly
point_cloud = get_dataset(42182).get_data(dataset_format='array')[0]


# or load lorenz dataset from pickle
point_cloud = point_net_utils.load_pickle('point_net_examples/lorenz.p')['lorenz']

# extract rho
rho = point_cloud[:,3]
# make a binary classification label based on the value fo rho
y = np.zeros(np.size(rho))
inds = np.argwhere(rho >= 10)
inds = np.squeeze(inds)
y[inds] = 1

# extract x, y, z of the lorenz
lorenz = point_cloud[:,[0,1,2]]


# window the lorenz into trajectories of fixed length
window_size = 32
window_stride = 8
SW = SlidingWindow(size=window_size, stride=window_stride)

X_sw, y_sw = SW.fit_transform_resample(lorenz, y)

# make train and test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_sw, y_sw)
# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

# Create Model
model = point_net.point_net(32, 3, units=32, dims=3)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=20, validation_data=test_dataset)


# save critical set examples
# high / low sets
low_samples = point_net_utils.select_samples(X_sw, y_sw, 100, 0)
high_samples = point_net_utils.select_samples(X_sw, y_sw, 100, 1)

low_preds = point_net.predict_critical(model, low_samples, layer_name = 'activation_14')
high_preds = point_net.predict_critical(model, high_samples, layer_name = 'activation_14')

low_cs, low_cs_mean = point_net.generate_critical(low_preds, 100, low_samples)
high_cs, high_cs_mean = point_net.generate_critical(high_preds, 100, high_samples)

low_cs_dict = {'low_cs': low_cs, 'low_cs_mean': low_cs_mean}
high_cs_dict = {'high_cs': high_cs, 'high_cs_mean': high_cs_mean}

point_net_utils.save_pickle(low_cs_dict, 'point_net_examples/trajectories/lorenz_low_cs.p')
point_net_utils.save_pickle(high_cs_dict, 'point_net_examples/trajectories/lorenz_high_cs.p')



# Random Shuffle Version

lorenz_s = np.copy(lorenz)
np.random.shuffle(lorenz_s)

window_size = 32
window_stride = 8
SW = SlidingWindow(size=window_size, stride=window_stride)

lor_sw = SW.fit_transform(lorenz)
lors_sw = SW.fit_transform(lorenz_s)

X_sw = np.vstack((lor_sw, lors_sw))

y = np.zeros(1228)
y[614:] = 1
y_sw = y
# make train and test sets
X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(X_sw, y_sw)
# Make tensors for the point net
train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

# Create Model
model = point_net.point_net(32, 3, units=32, dims=3)
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.02),metrics=['sparse_categorical_accuracy'])
# Fit Model
model.fit(train_dataset, epochs=20, validation_data=test_dataset)


# shuffle sets
true_samples = point_net_utils.select_samples(X_sw, y_sw, 100, 0)
shuffle_samples = point_net_utils.select_samples(X_sw, y_sw, 100, 1)

true_preds = point_net.predict_critical(model, true_samples, layer_name = 'activation_14')
shuffle_preds = point_net.predict_critical(model, shuffle_samples, layer_name = 'activation_14')

true_cs, true_cs_mean = point_net.generate_critical(true_preds, 100, true_samples)
shuffle_cs, shuffle_cs_mean = point_net.generate_critical(shuffle_preds, 100, shuffle_samples)

true_cs_dict = {'true_cs': true_cs, 'true_cs_mean': true_cs_mean}
shuffle_cs_dict = {'shuffle_cs': shuffle_cs, 'shuffle_cs_mean': shuffle_cs_mean}

point_net_utils.save_pickle(true_cs_dict, 'point_net_examples/trajectories/lorenz_true_cs.p')
point_net_utils.save_pickle(shuffle_cs_dict, 'point_net_examples/trajectories/lorenz_shuffle_cs.p')





# SCRIPT FOR GENERATING CCA TRAJECTORY SETS FOR RAW and CS SAMPLES
# start with samples and CS sets (generated from above scripts)
food_raw_examples, food_cs_examples, food_raw_aligned, food_cs_aligned = point_net_utils.generate_cca_trajectories(food_samples, food_cs, num_examples=3)
neut_raw_examples, neut_cs_examples, neut_raw_aligned, neut_cs_aligned = point_net_utils.generate_cca_trajectories(neut_samples, neut_cs, num_examples=3)
shock_raw_examples, shock_cs_examples, shock_raw_aligned, shock_cs_aligned = point_net_utils.generate_cca_trajectories(shock_samples, shock_cs, num_examples=3)
# select best matches from aligned trajectories    
food_raw_close = point_net_utils.select_closest_trajectories(food_raw_examples[0],food_raw_aligned[0],3)
food_cs_close = point_net_utils.select_closest_trajectories(food_cs_examples[0],food_cs_aligned[0],3)
food_raw_close2 = point_net_utils.select_closest_trajectories(food_raw_examples[1],food_raw_aligned[1],3)
food_cs_close2 = point_net_utils.select_closest_trajectories(food_cs_examples[1],food_cs_aligned[1],3)
food_raw_close3 = point_net_utils.select_closest_trajectories(food_raw_examples[2],food_raw_aligned[2],3)
food_cs_close3 = point_net_utils.select_closest_trajectories(food_cs_examples[2],food_cs_aligned[2],3)

neut_raw_close = point_net_utils.select_closest_trajectories(neut_raw_examples[0],neut_raw_aligned[0],3)
neut_cs_close = point_net_utils.select_closest_trajectories(neut_cs_examples[0],neut_cs_aligned[0],3)
neut_raw_close2 = point_net_utils.select_closest_trajectories(neut_raw_examples[1],neut_raw_aligned[1],3)
neut_cs_close2 = point_net_utils.select_closest_trajectories(neut_cs_examples[1],neut_cs_aligned[1],3)
neut_raw_close3 = point_net_utils.select_closest_trajectories(neut_raw_examples[2],neut_raw_aligned[2],3)
neut_cs_close3 = point_net_utils.select_closest_trajectories(neut_cs_examples[2],neut_cs_aligned[2],3)

shock_raw_close = point_net_utils.select_closest_trajectories(shock_raw_examples[0],shock_raw_aligned[0],3)
shock_cs_close = point_net_utils.select_closest_trajectories(shock_cs_examples[0],shock_cs_aligned[0],3)
shock_raw_close2 = point_net_utils.select_closest_trajectories(shock_raw_examples[1],shock_raw_aligned[1],3)
shock_cs_close2 = point_net_utils.select_closest_trajectories(shock_cs_examples[1],shock_cs_aligned[1],3)
shock_raw_close3 = point_net_utils.select_closest_trajectories(shock_raw_examples[2],shock_raw_aligned[2],3)
shock_cs_close3 = point_net_utils.select_closest_trajectories(shock_cs_examples[2],shock_cs_aligned[2],3)
# plot examples alongside best matches
fig = point_net_utils.plot_target_trajectory(food_raw_examples[0], food_raw_close, lines=True)
fig = point_net_utils.plot_target_trajectory(food_cs_examples[0], food_cs_close, lines=True)

fig = point_net_utils.plot_target_trajectory(neut_raw_examples[0], neut_raw_close, lines=True)
fig = point_net_utils.plot_target_trajectory(neut_cs_examples[0], neut_cs_close, lines=True)

fig = point_net_utils.plot_target_trajectory(shock_raw_examples[0], shock_raw_close, lines=True)
fig = point_net_utils.plot_target_trajectory(shock_cs_examples[0], shock_cs_close, lines=True)
# grid example
fig = point_net_utils.plot_target_trajectory_grid(food_raw_examples[1], food_raw_close2, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(food_cs_examples[0], food_cs_close, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(food_cs_examples[1], food_cs_close2, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(food_cs_examples[2], food_cs_close3, lines=True)

fig = point_net_utils.plot_target_trajectory_grid(neut_cs_examples[0], neut_cs_close, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(neut_cs_examples[1], neut_cs_close2, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(neut_cs_examples[2], neut_cs_close3, lines=True)

fig = point_net_utils.plot_target_trajectory_grid(shock_raw_examples[0], shock_raw_close, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(shock_raw_examples[1], shock_raw_close2, lines=True)
fig = point_net_utils.plot_target_trajectory_grid(shock_raw_examples[2], shock_raw_close3, lines=True)

# optional: assemble aligned raw and cs sets for TDA analysis
# TODO: remove zero row (not sure if it will make a difference)
X_raw_aligned = np.concatenate((neut_raw_aligned[0],food_raw_aligned[0],shock_raw_aligned[0]))
raw_aligned_length = np.shape(neut_raw_aligned[0])[0]
Y_raw_aligned = np.zeros(raw_aligned_length*3)
Y_raw_aligned[raw_aligned_length:raw_aligned_length*2] = 1
Y_raw_aligned[-raw_aligned_length:] = 2

X_cs_aligned = np.concatenate((neut_cs_aligned[0],food_cs_aligned[0],shock_cs_aligned[0]))
cs_aligned_neut_length = np.shape(neut_cs_aligned[0])[0]
cs_aligned_food_length = np.shape(food_cs_aligned[0])[0]
cs_aligned_shock_length = np.shape(shock_cs_aligned[0])[0]
Y_cs_aligned = np.zeros(cs_aligned_neut_length+cs_aligned_food_length+cs_aligned_shock_length)
Y_cs_aligned[cs_aligned_neut_length:cs_aligned_neut_length+cs_aligned_food_length] = 1
Y_cs_aligned[-cs_aligned_shock_length:] = 2

# optional: UMAP on aligned trajectories

# cs umap

# temp set size for critical sets (the subsample discards any that have fewer unique points)
test_size = 10

food_uniques, min_size_f = point_net_utils.cs_extract_uniques(food_cs_aligned[0], test_size)
neut_uniques, min_size_n = point_net_utils.cs_extract_uniques(neut_cs_aligned[0], test_size)
shock_uniques, min_size_s = point_net_utils.cs_extract_uniques(shock_cs_aligned[0], test_size)

food_sub_cs = point_net_utils.cs_subsample(food_uniques, test_size)
neut_sub_cs = point_net_utils.cs_subsample(neut_uniques, test_size)
shock_sub_cs = point_net_utils.cs_subsample(shock_uniques, test_size)

# Generate a PCA/UMAP projection of the whole set of samples(trajectories), 
all_sub_cs = np.concatenate((neut_sub_cs, food_sub_cs, shock_sub_cs))
Y_sub_cs = np.zeros(all_sub_cs.shape[0])
Y_sub_cs[len(neut_sub_cs):len(neut_sub_cs)+len(food_sub_cs)] = 1
Y_sub_cs[-len(shock_sub_cs):] = 2

# flatten for direct UMAP
X_flat_cs = np.reshape(all_sub_cs,(all_sub_cs.shape[0],-1))
mapper = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0)

X_flat_cs_umap = mapper.fit_transform(X_flat_cs)



# UPPER-BOUND SCRIPT
# load data, generate raw and CS aligned sets, subsample, etc.
food_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_013_food_cs.p')
neut_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_013_neut_cs.p')
shock_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_013_shock_cs.p')
food_samples = food_dict['food_samples']
food_cs = food_dict['food_cs']
neut_samples = neut_dict['neut_samples']
neut_cs = neut_dict['neut_cs']
shock_samples = shock_dict['shock_samples']
shock_cs = shock_dict['shock_cs']

test_size = 15
food_uniques, min_size_f = point_net_utils.cs_extract_uniques(food_cs, test_size)
neut_uniques, min_size_n = point_net_utils.cs_extract_uniques(neut_cs, test_size)
shock_uniques, min_size_s = point_net_utils.cs_extract_uniques(shock_cs, test_size)
food_sub_cs = point_net_utils.cs_subsample(food_uniques, test_size)
neut_sub_cs = point_net_utils.cs_subsample(neut_uniques, test_size)
shock_sub_cs = point_net_utils.cs_subsample(shock_uniques, test_size)

# direct to upper-bound shapes script
num_sets = 5
upper_size = 50
threshold = 0.2
neut_raw_uppers, neut_cs_uppers = point_net_utils.generate_upper_sets(neut_samples,np.array(neut_sub_cs), num_sets=num_sets, upper_size=upper_size, threshold=threshold)
food_raw_uppers, food_cs_uppers = point_net_utils.generate_upper_sets(food_samples,np.array(food_sub_cs), num_sets=num_sets, upper_size=upper_size, threshold=threshold)
shock_raw_uppers, shock_cs_uppers = point_net_utils.generate_upper_sets(shock_samples,np.array(shock_sub_cs), num_sets=num_sets, upper_size=upper_size, threshold=threshold)
# assemble into one list for TDA analysis
raw_uppers  = neut_raw_uppers + food_raw_uppers + shock_raw_uppers
cs_uppers = neut_cs_uppers + food_cs_uppers + shock_cs_uppers
n_labels = np.zeros(num_sets)
f_labels = np.ones(num_sets)
s_labels = np.ones(num_sets)*2
upp_labels = np.hstack((n_labels,f_labels,s_labels))


# old long-form version if you want the aligned trajectories directly
food_raw_examples, food_cs_examples, food_raw_aligned, food_cs_aligned = point_net_utils.generate_cca_trajectories(food_samples, np.array(food_sub_cs), num_examples=3)
neut_raw_examples, neut_cs_examples, neut_raw_aligned, neut_cs_aligned = point_net_utils.generate_cca_trajectories(neut_samples, np.array(neut_sub_cs), num_examples=3)
shock_raw_examples, shock_cs_examples, shock_raw_aligned, shock_cs_aligned = point_net_utils.generate_cca_trajectories(shock_samples, np.array(shock_sub_cs), num_examples=3)

# approximate upper-bound generation
# take a set of aligned CS (sub-sample to keep runtime down) and take all unique points (within threshold)
# across the set of CS
# then plot this as a 3d shape
# neut
ind_arr = np.arange(neut_cs_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
neut_cs_sub_align = neut_cs_aligned[0][rand_inds,:,:]
point_list_neut, point_arr_neut = point_net_utils.generate_uniques_from_trajectories(neut_cs_examples[0],neut_cs_sub_align, threshold=0.2)
# food
ind_arr = np.arange(food_cs_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
food_cs_sub_align = food_cs_aligned[0][rand_inds,:,:]
point_list_food, point_arr_food = point_net_utils.generate_uniques_from_trajectories(food_cs_examples[0],food_cs_sub_align, threshold=0.2)
# shock
ind_arr = np.arange(shock_cs_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
shock_cs_sub_align = shock_cs_aligned[0][rand_inds,:,:]
point_list_shock, point_arr_shock = point_net_utils.generate_uniques_from_trajectories(shock_cs_examples[0],shock_cs_sub_align, threshold=0.2)
# plotting using pyvista
import pyvista as pv
# example surface volume plot
cloud = pv.PolyData(point_arr_shock)
cloud.plot(notebook=False)
volume = cloud.reconstruct_surface()
volume.plot(notebook=False)
# seems redundant on this data
shell = volume.extract_geometry()
smooth = shell.smooth(n_iter=10)
smooth.plot(notebook=False)
# example voxel plot (needs closed mesh...TODO)
voxels = pv.voxelize(volume)
p = pv.plotter(notebook=False)
p.add_mesh(voxels, color=True, show_edges=True)

# surface plots including loading from pickle
import surface
cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/alignment_samples/upper_bound_cs_examples.p')
raw_dict = point_net_utils.load_pickle('point_net_examples/trajectories/alignment_samples/upper_bound_raw_examples.p')
upper_raw_neut = raw_dict['upper_raw_neut']
upper_raw_food = raw_dict['upper_raw_food']
upper_raw_shock = raw_dict['upper_raw_shock']
upper_cs_neut = cs_dict['upper_cs_neut']
upper_cs_food = cs_dict['upper_cs_food']
upper_cs_shock = cs_dict['upper_cs_shock']
upper_raw = np.vstack((upper_raw_neut,upper_raw_food,upper_raw_shock))
upper_cs = np.vstack((upper_cs_neut,upper_cs_food,upper_cs_shock))

upper_raw_labels = np.zeros(upper_raw.shape[0])
upper_raw_labels[upper_raw_neut.shape[0]:upper_raw_food.shape[0]+upper_raw_neut.shape[0]] = 1
upper_raw_labels[-upper_raw_shock.shape[0]:] = 2

upper_cs_labels = np.zeros(upper_cs.shape[0])
upper_cs_labels[upper_cs_neut.shape[0]:upper_cs_food.shape[0]+upper_cs_neut.shape[0]] = 1
upper_cs_labels[-upper_cs_shock.shape[0]:] = 2
surface.pv_voxel_ovr_plot(upper_raw,upper_raw_labels)
surface.pv_voxel_ovr_plot(upper_cs,upper_cs_labels)



# as above, but with raw trajectories instead of CS
# neut
ind_arr = np.arange(neut_raw_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
neut_raw_sub_align = neut_raw_aligned[0][rand_inds,:,:]
point_list_rneut, point_arr_rneut = point_net_utils.generate_uniques_from_trajectories(neut_raw_examples[0],neut_raw_sub_align, threshold=0.2)
# food
ind_arr = np.arange(food_raw_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
food_raw_sub_align = food_raw_aligned[0][rand_inds,:,:]
point_list_rfood, point_arr_rfood = point_net_utils.generate_uniques_from_trajectories(food_raw_examples[0],food_raw_sub_align, threshold=0.2)
# shock
ind_arr = np.arange(shock_raw_aligned[0].shape[0])
rand_inds = np.random.choice(ind_arr,100,replace=False)
shock_raw_sub_align = shock_raw_aligned[0][rand_inds,:,:]
point_list_rshock, point_arr_rshock = point_net_utils.generate_uniques_from_trajectories(shock_raw_examples[0],shock_raw_sub_align, threshold=0.2)

# plotting using open3d
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_arr_rshock)
o3d.visualization.draw_geometries([pcd])

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.1)
o3d.visualization.draw_geometries([voxel_grid])





# Critical Sets Statistics Script

# load context cs
neut_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_neut_cs.p')
food_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_food_cs.p')
shock_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_shock_cs.p')
# load behaviour cs
b31n_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_b31n_cs.p')
b08f_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_b08f_cs.p')
b07s_cs_dict = point_net_utils.load_pickle('point_net_examples/trajectories/session_0134_b07s_cs.p')

# extract uniques from CS
food_all_cs = food_cs_dict['food_cs']
neut_all_cs = neut_cs_dict['neut_cs']
shock_all_cs = shock_cs_dict['shock_cs']
food_uniques, min_size_f = point_net_utils.cs_extract_uniques(food_all_cs, food_cs_dict['food_cs_mean'])
neut_uniques, min_size_n = point_net_utils.cs_extract_uniques(neut_all_cs, neut_cs_dict['neut_cs_mean'])
shock_uniques, min_size_s = point_net_utils.cs_extract_uniques(shock_all_cs, shock_cs_dict['shock_cs_mean'])

b08f_all_cs = b08f_cs_dict['food_cs']
b31n_all_cs = b31n_cs_dict['neut_cs']
b07s_all_cs = b07s_cs_dict['shock_cs']
b08f_uniques, min_size_b08f = point_net_utils.cs_extract_uniques(b08f_all_cs, b08f_cs_dict['food_cs_mean'])
b31n_uniques, min_size_b31n = point_net_utils.cs_extract_uniques(b31n_all_cs, b31n_cs_dict['neut_cs_mean'])
b07s_uniques, min_size_b07s = point_net_utils.cs_extract_uniques(b07s_all_cs, b07s_cs_dict['shock_cs_mean'])

# extract sizes of uniques
food_lengths = np.array([i.shape[0] for i in food_uniques])
neut_lengths = np.array([i.shape[0] for i in neut_uniques])
shock_lengths = np.array([i.shape[0] for i in shock_uniques])

b08f_lengths = np.array([i.shape[0] for i in b08f_uniques])
b31n_lengths = np.array([i.shape[0] for i in b31n_uniques])
b07s_lengths = np.array([i.shape[0] for i in b07s_uniques])    

# calc strandard errors
scipy.stats.sem(food_lengths)
#Out[37]: 0.21725814392353882
scipy.stats.sem(shock_lengths)
#Out[38]: 0.1733695705941732
scipy.stats.sem(neut_lengths)
#Out[39]: 0.1514356753336669
scipy.stats.sem(b31n_lengths)
#Out[40]: 0.34280335677198587
scipy.stats.sem(b08f_lengths)
#Out[41]: 0.23331385200059068
scipy.stats.sem(b07s_lengths)
#Out[42]: 0.30887707992832764

# t test 
ttest_ns = scipy.stats.ttest_ind(neut_lengths, shock_lengths, equal_var=False, trim=0.1)
ttest_fs = scipy.stats.ttest_ind(food_lengths, shock_lengths, equal_var=False, trim=0.1)
ttest_fn = scipy.stats.ttest_ind(food_lengths, neut_lengths, equal_var=False, trim=0.1)

ttest_b31b07 = scipy.stats.ttest_ind(b31n_lengths, b07s_lengths, equal_var=False)
ttest_b08b07 = scipy.stats.ttest_ind(b08f_lengths, b07s_lengths, equal_var=False)
ttest_b08b31 = scipy.stats.ttest_ind(b08f_lengths, b31n_lengths, equal_var=False)






# TEMP Investigating windowed neural data for context prototypes
from sklearn.metrics import pairwise_distances
# start with windowed data: X_sw, Y_sw

food_inds = Y_sw == 1
shock_inds = Y_sw == 2
neut_inds = Y_sw == 0

food_sw = X_sw[food_inds,:,:]
shock_sw = X_sw[shock_inds,:,:]
neut_sw = X_sw[neut_inds,:,:]

food_flat = np.reshape(food_sw, (food_sw.shape[0],-1))
shock_flat = np.reshape(shock_sw, (shock_sw.shape[0],-1))
neut_flat = np.reshape(neut_sw, (neut_sw.shape[0],-1))

# can change metric if needed
food_dists = pairwise_distances(food_flat)
shock_dists = pairwise_distances(shock_flat)
neut_dists = pairwise_distances(neut_flat)

# grab lower half of diagonal matrix
food_lower = np.tril(food_dists, k=-1)
shock_lower = np.tril(shock_dists, k=-1)
neut_lower = np.tril(neut_dists, k=-1)

# set 0s to 100, so we can use min selection
food_lower[food_lower==0] = 100
shock_lower[shock_lower==0] = 100
neut_lower[neut_lower==0] = 100

# select min per row
food_mins = food_lower.min(axis=1)
shock_mins = shock_lower.min(axis=1)
neut_mins = neut_lower.min(axis=1)

# get the indices of the 10 lowest values
food_minds = np.argpartition(food_mins, 10)[:10]
shock_minds = np.argpartition(shock_mins, 10)[:10]
neut_minds = np.argpartition(neut_mins, 10)[:10]

# grab those 10 trajectories from each context
food_protos = food_sw[food_minds,:,:]
shock_protos = shock_sw[shock_minds,:,:]
neut_protos = neut_sw[neut_minds,:,:]


# project with pca for plotting
food_pca_cs, food_pca_samps = point_net_utils.pca_cs_windowed(food_protos, food_sw, dims=3)
shock_pca_cs, shock_pca_samps = point_net_utils.pca_cs_windowed(shock_protos, shock_sw, dims=3)
neut_pca_cs, neut_pca_samps = point_net_utils.pca_cs_windowed(neut_protos, neut_sw, dims=3)

fig = point_net_utils.plot_samples(food_pca_cs, 5)
fig = point_net_utils.plot_samples(shock_pca_cs, 5)
fig = point_net_utils.plot_samples(neut_pca_cs, 5)


# project with umap for plotting
neut_umap_cs, neut_umap_samps = point_net_utils.umap_cs_windowed(neut_protos, neut_sw, dims=3)
food_umap_cs, food_umap_samps = point_net_utils.umap_cs_windowed(food_protos, food_sw, dims=3)
shock_umap_cs, shock_umap_samps = point_net_utils.umap_cs_windowed(shock_protos, shock_sw, dims=3)



# temp functions for getting distances and plotting NOT NECESSARY CURRENTLY

import matplotlib.patheffects as PathEffects

def generate_distance_matrix(data, labels, n_clusters, metric='euclidean'):

    avg_dists = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            if data[labels==i].any() and data[labels==j].any():
                avg_dists[i,j] = pairwise_distances(data[labels==i],data[labels==j], metric=metric).mean()
            else:
                avg_dists[i,j] = -1
    avg_dists /= avg_dists.max()
    avg_dists[avg_dists < 0] = -1
    return avg_dists

# function to plot a single distance matrix
def plot_distance_matrix(avg_dists, n_clusters, show_values=False):
    fig = plt.figure()
    if show_values:
        for i in range(n_clusters):
            for j in range(n_clusters):
                t = plt.text(i,j, '%5.3f' % avg_dists[i,j], verticalalignment='center', horizontalalignment='center')
                t.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w', alpha=0.5)])

    plt.imshow(avg_dists, interpolation='nearest', cmap='viridis')
    plt.xticks(range(n_clusters), range(n_clusters), rotation=45)
    plt.yticks(range(n_clusters), range(n_clusters))
    plt.colorbar()
    plt.tight_layout()

    return fig
    




