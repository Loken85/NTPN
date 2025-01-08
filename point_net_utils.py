#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:47:28 2023

Utility Functions for use with the Point Net used in conjunction with binned spike counts and labels


@author: proxy_loken
"""

import numpy as np
import pickle

import tensorflow as tf
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import StandardScaler
from gtda.time_series import SlidingWindow
from sklearn.model_selection import train_test_split

import umap
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection




# Pickling/Unpickling Data
def save_pickle(data, filename):
    
    pickle.dump(data, open(filename, 'wb'))
    


def load_pickle(filename):    
    with open(filename,'rb') as fp:
        l_data = pickle.load(fp)
        
    return l_data



# DATA LOADING FUNCTIONS

# Function to load dataset and labelset from pickle
# Built to work with the current hubdt output format. Will need to be changed for other data formats
# TODO: branch to accomodate multilabelset loading
def load_data_pickle(st_file, label_file, label_name):
    
    st_dict = load_pickle(st_file)
    stbin_list = st_dict['raw_stbins']
    label_dict = load_pickle(label_file)
    label_list = label_dict[label_name]
    
    return stbin_list, label_list




# SAMPLING / SUBSAMPLING DATA FUNCTIONS

# function to extract a set of samples rom a given class for testing/visualisation
# INPUT: X: data array(3D, select along axis=0), Y: class labels,  num_samples: number of samples to extract (should match batch size),
# class_label: label of class to extract
def select_samples(X, Y, num_samples, class_label, return_index=False):
    
    inds = np.argwhere(Y == class_label)
    inds = np.squeeze(inds)
    selection = np.random.choice(inds, num_samples, False)
    samples = X[selection,:,:]
    if return_index:
        return samples, selection
    return samples


# function to extract a random subset of samples and accompanying critical sets
def select_samples_cs(cs, samples, num_samples, return_index=False):
    
    inds = np.arange(samples.shape[0])
    selection = np.random.choice(inds, num_samples, False)
    samples_out = samples[selection,:,:]
    cs_out = cs[selection,:,:]
    if return_index:
        return cs_out, samples_out, selection
    return cs_out, samples_out



# subsample neurons within each session, then stack each session and labels into a single dataset
def subsample_dataset_3d_within(stbin_list, labels_list, sample_size=32, replace=True):
    
    out_list = []
    for i in range(len(stbin_list)):
        out = subsample_neurons_3d(stbin_list[i], sample_size, replace)
        out_list.append(out)
        
        out_neurons = np.concatenate(out_list)
        out_labels = np.concatenate(labels_list)
        
    return out_neurons, out_labels


# subsample neurons across sessions. Assemble a dataset of fixed size (per label)
# Assumes the labels for each session contain the same set of labels
def subsample_dataset_3d_across(stbin_list, labels_list, num_samples=2000, sample_size=32, replace=True):
    
    labels = np.unique(labels_list[0])
    all_neurons = []
    all_labels = []
    for i in labels:
        # select all samples for each label from all sessions
        curr_neurons = []
        for j in range(len(stbin_list)):
            curr_samples = select_samples(stbin_list[j], labels_list[j], num_samples, i)
            curr_neurons.append(curr_samples)
        
        curr_neurons = np.concatenate(curr_neurons, axis=1)
        all_neurons.append(curr_neurons)
        all_labels.append(np.zeros(num_samples)+i)
    
    all_neurons = np.concatenate(all_neurons)
    all_labels = np.concatenate(all_labels)
    
    out_neurons = subsample_neurons_3d(all_neurons, sample_size, replace)    
    
    return out_neurons, all_labels


def subsample_neurons(stbin, sample_size=32, replace=True):
    
    ind_arr = np.arange(np.size(stbin,1))
    
    rand_inds = np.random.choice(ind_arr, sample_size, replace)
    
    out_neurons = stbin[:,rand_inds].copy()
    
    return out_neurons


def subsample_neurons_3d(stbin, sample_size=32, replace=True):
    
    ind_arr = np.arange(np.size(stbin,1))
    
    rand_inds = np.random.choice(ind_arr, sample_size, replace)
    
    out_neurons = stbin[:,rand_inds,:].copy()
    
    return out_neurons



# DATA PERMUTATION / AUGMENTATION


def gen_permuted_data(neurons, labels, direction='width', samples=10):
    
    out_neurons = []
    
    if direction == 'width':
        for i in range(samples):
            sample_neurons = subsample_neurons(neurons, sample_size=32, replace=True)
            out_neurons.append(sample_neurons)
        
        neuron_array = np.hstack(out_neurons)
        labels_array = labels
    
    elif direction == 'length':
        for i in range(samples):
            sample_neurons = subsample_neurons(neurons, sample_size=32, replace=True)
            out_neurons.append(sample_neurons)
            
        neuron_array = np.vstack(out_neurons)
        labels_array = np.repeat(labels, samples, axis=0)
        
    return neuron_array, labels_array
            



# Function to augment data points by adding jitter
# TODO
def augment(points, label):
    # TODO change params for neural data
    points += tf.random.uniform(points.shape, -0.05, 0.05, dtype=tf.float64)
    
    points = tf.random.shuffle(points)    
    
    return points, label


# function to generate a unit sphere of points to test upper bound shapes for classes
def unit_sphere():
    
    lim = 0.577
    all_points = []
    
    for x in np.linspace(-lim,lim, 16):
        for y in np.linspace(-lim,lim, 16):
            for z in np.linspace(-lim,lim, 16):
                all_points.append([x, y, z])
    
    all_points = np.array(all_points)
    # reshape to match input dims
    all_points = np.reshape(all_points,(512,32,3))
    
    
    return all_points



# DATA PREPROCESSING / TRANSFORMS


# Helper pre-processing function to cut out noise periods from labelsets and
# accompanying numerical data
# labels and num_data need to be the same length, asumes labels are a vector
# assumes noise is labelled as -1 in labels
def precut_noise(labels, num_data, noise_label=-1):
    
    # get mask of non-noise labels
    mask = labels != noise_label
    outlabels = labels[mask].copy()
    
    # check for dim of numerical data
    if num_data.ndim == 1:
        outdata = num_data[mask].copy()
    else:
        outdata = num_data[mask,:].copy()
    
    return outlabels, outdata


# Remove noise from the portion of the dataset/labels in selection
# selection should be a list of indices of sessions to include
def remove_noise_cat(stbin_list, label_list, selection, noise_label=-1):
    
    if len(selection)<=1:
        label_cut, st_cut = precut_noise(label_list, stbin_list, noise_label)
        return st_cut, label_cut
    
    stcut_list = []
    label_cut_list = []
    for i in selection:
        label_cut, st_cut = precut_noise(label_list[i], stbin_list[i], noise_label)
        stcut_list.append(st_cut)
        label_cut_list.append(label_cut)
        
    
    return stcut_list, label_cut_list


def pow_transform(stbin_list, selection):
    
    if len(selection)<=1:
        X_pow = power_transform(stbin_list)
        return X_pow
    
    X_pow_list = []
    for i in selection:
        X_pow_list.append(power_transform(stbin_list[i]))
    
    return X_pow_list


def std_transform(stbin_list, selection):
    
    if len(selection)<=1:
        X_std = StandardScaler().fit_transform(stbin_list)
        return X_std
    
    X_std_list = []
    for i in selection:
        std = StandardScaler()
        X_std = std.fit_transform(stbin_list[i])
        X_std_list.append(X_std)
        
    return X_std_list


# Project the 1-d time series of stbin into an N-d time series by windowing adjecent time bins, also adjusts the labels
# The window size is equivalent to the output dimension, ie a window of 3 produces a '3D' projection
# stride can be adjusted to subsample during the projection (stride of 1 will have N-1 overlap between windows)
def window_projection(stbin_list, label_list, selection, window_size=3, stride=1):
    
    if len(selection)<=1:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(stbin_list, label_list)
        return X_sw, Y_sw
    
    X_sw_list = []
    Y_sw_list = []
    for i in selection:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(stbin_list[i], label_list[i])
        X_sw = np.swapaxes(X_sw,1,2)
        X_sw_list.append(X_sw)
        Y_sw_list.append(Y_sw)
        
    return X_sw_list, Y_sw_list


# helper for behavioural segmentation. Windows the behavioural labels to match the stbin windowing in the function above
def window_projection_segments(behav_list, label_list, selection, window_size=3, stride=1):
    
    if len(selection)<=1:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(behav_list, label_list)
        return X_sw, Y_sw
    
    X_sw_list = []
    Y_sw_list = []
    for i in selection:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(behav_list[i], label_list[i])
        
        X_sw_list.append(X_sw)
        Y_sw_list.append(Y_sw)
        
    return X_sw_list, Y_sw_list




# DATASET SPLITTING

# Helper to call sklearn's train_test_split to generate training and test matrices
def train_test_gen(Xs, Ys, test_size=0.2, stratify=False):
    
    if stratify:
        X_train, X_val, Y_train, Y_val = train_test_split(Xs, Ys, test_size=test_size, stratify=Ys)
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(Xs, Ys, test_size=test_size)
    
    return X_train, X_val, Y_train, Y_val


# helper to convert train, test datasets into tensorflow tensors for use in NN training/testing
# uses tensorflow functions to convert to tensors, batch, and augment the training set (optional)
def train_test_tensors(X_train, X_val, Y_train, Y_val, augment=True, batch_size=8):
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    # augment and shuffle
    # TODO: update augmentation
    if augment:
        train_dataset = train_dataset.shuffle(len(X_train)).map(augment).batch(batch_size, drop_remainder=True)
    else:
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(len(X_val)).batch(batch_size, drop_remainder=True)
    
    return train_dataset, test_dataset


def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train,:,:]
    X_test = data[ix_test,:,:]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test




# UMAP Dimensionality reduction functions

# function for mapping windowed samples into reduceed dimension using UMAP
def umap_windowed(data, dims=2, neighbours=30, m_dist=0.0):
    
    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(data)
    flat_data = np.reshape(data, (-1, shape[2]))
    
    u_model = umap.UMAP(n_neighbors=neighbours, n_components=dims, min_dist=m_dist).fit(flat_data)    
    
    # transform samples
    mapped_data = np.zeros((shape[0],shape[1],dims))
    for i in range(shape[0]):
        mapped_data[i,:,:] = u_model.transform(data[i,:,:])    
    
    return mapped_data


# function for mapping windowed critical sets and samples into reduced dimension using UMAP
def umap_cs_windowed(cs, samples, dims=2, neighbours=30, m_dist=0.0):
    
    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(samples)
    flat_data = np.reshape(samples, (-1, shape[2]))
    
    u_model = umap.UMAP(n_neighbors=neighbours, n_components=dims, min_dist=m_dist).fit(flat_data)    
    
    # transform samples
    mapped_samples = np.zeros((shape[0],shape[1],dims))
    for i in range(shape[0]):
        mapped_samples[i,:,:] = u_model.transform(samples[i,:,:])   
    
    # transform cs
    cs_shape = np.shape(cs)
    mapped_cs = np.zeros((cs_shape[0], cs_shape[1], dims))
    for i in range(cs_shape[0]):
        mapped_cs[i,:,:] = u_model.transform(cs[i,:,:])
    
    return mapped_cs, mapped_samples


# function for mapping windowed critical sets and samples into reduced dimension using PCA
def pca_cs_windowed(cs, samples, dims=2):
    
    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(samples)
    flat_data = np.reshape(samples, (-1, shape[2]))
    
    pca_model = PCA(n_components=dims).fit(flat_data)
    
    # transform samples
    mapped_samples = np.zeros((shape[0],shape[1],dims))
    for i in range(shape[0]):
        mapped_samples[i,:,:] = pca_model.transform(samples[i,:,:])   
    
    # transform cs
    cs_shape = np.shape(cs)
    mapped_cs = np.zeros((cs_shape[0], cs_shape[1], dims))
    for i in range(cs_shape[0]):
        mapped_cs[i,:,:] = pca_model.transform(cs[i,:,:])
    
    return mapped_cs, mapped_samples


# CS Processing

# function for extracting unique points from critical set trajectories
# critical sets will contian many duplicates, this will return an in order set
# of unique points
# INPUTS: cs - critical sets (samples x points x neurons), cs_mean - mean unique count
# OUTPUTS: uniques_list - list of unique points for each sample, min_size - smallest unique point set
def cs_extract_uniques(cs, cs_mean):
    
    uniques_list = []
    min_size = cs_mean
    for i in np.arange(np.shape(cs)[0]):
    
        unique_points, indxs = np.unique(cs[i,:,:], return_index=True, axis=0)
        unique_order = cs[i,np.sort(indxs),:]
        min_size = np.min([np.shape(unique_order)[0], min_size])
        uniques_list.append(unique_order)
    
    return uniques_list, min_size
    
# function for subsampling critical set lists to min_size
# skips samples that have less than min_size points
def cs_subsample(cs_list, min_size):
    
    subs_list = []
    for i in np.arange(len(cs_list)):
        if np.shape(cs_list[i])[0] < min_size:
            continue
        uniques_sub = subsample_neurons(cs_list[i].T,sample_size=min_size,replace=False)
        subs_list.append(uniques_sub.T)
    
    return subs_list


# TRAJECTORY ALIGNMENT FUNCTIONS

# function to generate aligned trajectories for each of num_examples raw samples and critical sets
# returns a list of aligned trajectories matrices (one for each example) as well as the original examples 
def generate_cca_trajectories(samples, cs, num_examples=5):
    
    # select num_examples random entries from samples/cs
    inds = np.arange(cs.shape[0])
    selection = np.random.choice(inds, num_examples, False)
    samples_out = samples[selection,:,:]
    cs_out = cs[selection,:,:]    
    
    # for each example, generate a matrix of CCA aligned trajectories (all other samples/cs)
    samples_aligned_trajs = []
    cs_aligned_trajs = []
    samples_examples = []
    cs_examples = []
    for i in range(len(selection)):
        sample_example, sample_aligned = calc_cca_trajectories(samples_out[i], samples, selection[i])
        cs_example, cs_aligned = calc_cca_trajectories(cs_out[i], cs, selection[i])
        samples_aligned_trajs.append(sample_aligned)
        cs_aligned_trajs.append(cs_aligned)
        samples_examples.append(sample_example)
        cs_examples.append(cs_example)
    
    
    return samples_examples, cs_examples, samples_aligned_trajs, cs_aligned_trajs


# helper to calculate a set of aligned trajectories to an exemplar using cca
def calc_cca_trajectories(exemplar, samples, ex_index, ndims=3):
    
    aligned_trajs = np.zeros((samples.shape[0],samples.shape[1],ndims))
    for i in range(samples.shape[0]):
        if i == ex_index:
            # set to an arbitrary value to be ignored in later analysis
            aligned_trajs[i,:,:] = 10
            continue
        cca = CCA(n_components=ndims)
        aligned_trajs[i,:,:], example = cca.fit_transform(samples[i], exemplar)
        
    return example, aligned_trajs


# function to select num_examples closest (by distance) algined trajectories to the exemplar trajectory
# returns those trajectories
def select_closest_trajectories(exemplar, aligned_trajectories, num_examples):
    # calculate the distances between the examplar and all algined trajectories
    dists = pairwise_distances(np.reshape(aligned_trajectories, (aligned_trajectories.shape[0],-1)),np.reshape(exemplar,(1,-1)))
    dists = np.squeeze(dists)
    # get the indices of the num_examples smallest distances
    inds = np.argpartition(dists,num_examples)[:num_examples]
    # select those indices from the aligned_trajectories array
    select_trajs = aligned_trajectories[inds]
    
    return select_trajs



# function to generate a set of unique(or approximately unique via threshold) points from a set of aligned trajectories
# mode: fixed - takes a manually defined threshold, dynamic - calculates a threshold based on the statistics of the trajectories
# returns a list of those unique points
def generate_uniques_from_trajectories(exemplar, trajectories, mode='fixed', threshold=0.0):
    
    point_set = []
    traj_flat = np.reshape(trajectories.copy(),(-1,trajectories.shape[2]))
    all_points = np.concatenate((exemplar, traj_flat))
    #overlap = np.ones(len(all_points), dtype=bool)
    if mode == 'dynamic':
        # TODO: set threshold calculation to something sensible
        threshold = np.std(all_points)
    
    i=0
    while i < len(all_points)-1:
        # check for dummy values to be removed TODO: change the dummy value to nan or somethign else
        if all_points[i,0] == 10:
            np.delete(all_points,i,axis=0)
            continue
        # calculate distances between current point and remaining points in the set
        dists = pairwise_distances(all_points[i+1:,:],np.reshape(all_points[i,:],(1,-1)))
        # get indices of dists that are below threshold
        overlap = np.argwhere(np.squeeze(dists) <= threshold)
        # correct overlap indices to apply to all_points array
        overlap = overlap + i
        # remove points from the set based on the overlap indices
        all_points = np.delete(all_points, overlap, axis=0)
        # add current point to point_set list (sanity check: point_set and all_points should match at end)
        point_set.append(all_points[i])
        # increment i
        i += 1  
    
    
    
    return point_set, all_points


# all in one function to generate upper-bound sets using aligned(by CCA) CS
# INPUTS: samples - raw trajectories, cs - critical sets (already subsampled ideally), num_sets - number of uppers to create
# upper_size - number of trajectories to build upper-bound shape from, threshold - distance for pruning points
def generate_upper_sets(samples, cs, num_sets=3, upper_size=50, threshold = 0.2):
    
    raw_examples, cs_examples, raw_aligned, cs_aligned = generate_cca_trajectories(samples, cs, num_examples = num_sets)
    
    raw_uppers = []
    cs_uppers = []
    
    for i in range(num_sets):
        ind_arr = np.arange(raw_aligned[i].shape[0])
        rand_inds = np.random.choice(ind_arr, upper_size, replace=False)
        raw_aligned_sub = raw_aligned[i][rand_inds,:,:]
        point_list, point_arr = generate_uniques_from_trajectories(raw_examples[i],raw_aligned_sub, threshold=threshold)
        raw_uppers.append(point_arr)
        ind_arr = np.arange(cs_aligned[i].shape[0])
        rand_inds = np.random.choice(ind_arr, upper_size, replace=False)
        cs_aligned_sub = cs_aligned[i][rand_inds,:,:]
        point_list, point_arr = generate_uniques_from_trajectories(cs_examples[i],cs_aligned_sub, threshold=threshold)
        cs_uppers.append(point_arr)
    
    return raw_uppers, cs_uppers




# VISUALISATION FUNCTIONS (Plotting to accompany layer visualisation generation)

# helper for creating a 3d line collection from a trajectory for plotting
def trajectory_to_lines(traj, width=2, alpha=1, cmap='viridis'):
    
    # reshape trajectory to be (numlines) x (points per line) x (dims: x, y, z)
    points = traj.copy()
    points = points.reshape(-1,1,3)
    # then concat into segments
    segments = np.concatenate([points[:-1],points[1:]],axis=1)
    # generate a norm for the range of values for colour (0:length) if using index
    norm = plt.Normalize(0,traj.shape[0])
    lines = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
    # set the values for colour mapping
    lines.set_array(np.arange(traj.shape[0]))
    lines.set_linewidth(width)
    
    return lines



# helper for creatnig a 3d line collection from an unordered point cloud for plotting
def cloud_to_lines(cloud, width=2, alpha=1, cmap='viridis'):
    # reshape point cloud to be (numlines) x (points per line) x (dims: x, y, z)
    points = cloud.copy()
    points = points.reshape(-1,1,3)
    # then concat into segments
    segments = np.concatenate([points[:-1],points[1:]],axis=1)
    lines = Line3DCollection(segments, cmap=cmap, alpha=alpha)
    lines.set_linewidth(width)
    
    return lines



# plot select samples
# TODO: add error on missing mode return
def plot_samples(samples, num_samples, mode='scatter'):
    
    fig, axs = plt.subplots(num_samples, subplot_kw=dict(projection='3d'))
    col_map = plt.get_cmap('viridis')
    
    for i in range(num_samples):
        xs = samples[i][:,0]
        ys = samples[i][:,1]
        zs = samples[i][:,2]
        c = np.arange(samples.shape[1])
        if mode == 'scatter':
            sax = axs[i].scatter(xs, ys, zs, c=c, cmap=col_map)
            fig.colorbar(sax, ax=axs)
        elif mode == 'line':
            c = np.linspace(0,1,samples.shape[1])
            # zip version, deprecated
            #for begin, end in zip(samples[i][:-1], samples[i][1:]):
                #x, y, z = zip(begin, end)
                #colour = c[i]
                #sax = axs[i].plot(x,y,z, c=f'C{c[i]}')
            for j in range(samples.shape[1]-1):                
                sax = axs[i].plot(xs[j:j+2],ys[j:j+2],zs[j:j+2], c=plt.cm.viridis(c[j]))
                #norm = plt.Normalize(0,samples.shape[1])
                #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=col_map), ax = axs)
            
        else:
            return 0
        
    #fig.colorbar(sax, ax = axs)
        
            
    return fig


# plot a single sample as a 3d scatter or line, coloured by index in the trajectory
def plot_sample(sample, mode='scatter', trajectory=True):
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if mode == 'scatter':
        xs = sample[:,0]
        ys = sample[:,1]
        zs = sample[:,2]
        if trajectory:
            c = np.arange(sample.shape[0])
            sax = ax.scatter(xs,ys,zs, c=c, cmap='viridis')
            fig.colorbar(sax, ax=ax)
        else:
            c = zs
            sax = ax.scatter(xs,ys,zs,c=c,cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])    
        
    elif mode == 'line':
        if trajectory:
            lc = trajectory_to_lines(sample, width=2, cmap='viridis')
        else:
            lc = cloud_to_lines(sample, width=2, cmap='viridis')
        lines = ax.add_collection(lc)
        ax.set_xlim(np.min(sample[:,0]),np.max(sample[:,0]))
        ax.set_ylim(np.min(sample[:,1]),np.max(sample[:,1]))
        ax.set_zlim(np.min(sample[:,2]),np.max(sample[:,2]))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if trajectory:
            fig.colorbar(lines, ax = ax)
    else:
        return 0
    
    return fig
    

# plot a segmented sample with each point labelled
def plot_sample_segmented(sample, labels, remove_noise=False):
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if remove_noise:
        mask = labels != -1
        sample=sample[mask]
        labels = labels[mask]
    
    
    xs = sample[:,0]
    ys = sample[:,1]
    zs = sample[:,2]
    
    c = labels
    
    sax=ax.scatter(xs,ys,zs,c=c, cmap='viridis')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    fig.colorbar(sax, ax=ax)
    
    return fig


    
# plot a reference trajectory with comparative trajectories as overlapping points
def plot_target_trajectory(target, comps, lines=False):
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))    
    # get line collection for target trajectory
    lc = trajectory_to_lines(target, width=2, cmap='viridis')
    lines = ax.add_collection(lc)
    ax.set_xlim(np.min(target[:,0]),np.max(target[:,0]))
    ax.set_ylim(np.min(target[:,1]),np.max(target[:,1]))
    ax.set_zlim(np.min(target[:,2]),np.max(target[:,2]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(lines, ax = ax)
    
    if np.ndim(comps) == 2:
        if lines:
            clc = trajectory_to_lines(comps, width=5, alpha=0.3, cmap='viridis')
            clines = ax.add_collection(clc)
        xs = comps[:,0]
        ys = comps[:,1]
        zs = comps[:,2]
        c = np.arange(comps.shape[0])
        sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis',alpha=0.6)
            
    else:
        for i in range(comps.shape[0]):
            if lines:
                clc = trajectory_to_lines(comps[i], width=5, alpha=0.3, cmap='viridis')
                clines = ax.add_collection(clc)
            xs = comps[i][:,0]
            ys = comps[i][:,1]
            zs = comps[i][:,2]
            c = np.arange(comps.shape[1])
            sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis',alpha=0.6)
    plt.show()
    return fig
            

# plot a reference trajectory alongside comparative trajectories
def plot_target_trajectory_grid(target, comps, lines=False):
    
    fig = plt.figure()
    num_comps = np.shape(comps)[0]
    # define gridspec for figure
    gs = fig.add_gridspec(1,1, right=0.65)
    # define main axis
    ax = fig.add_subplot(gs[0,0], projection='3d')
    lc = trajectory_to_lines(target, width=2, cmap='viridis')
    lines = ax.add_collection(lc)
    ax.set_xlim(np.min(target[:,0]),np.max(target[:,0]))
    ax.set_ylim(np.min(target[:,1]),np.max(target[:,1]))
    ax.set_zlim(np.min(target[:,2]),np.max(target[:,2]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(lines, ax = ax, location='left')    
    
    gs2 = fig.add_gridspec(num_comps,1,left=0.7)
    for i in range(num_comps):
        ax_comp = fig.add_subplot(gs2[i,0], projection='3d')
        if lines:
            clc = trajectory_to_lines(comps[i], width=3, alpha=0.6, cmap='viridis')
            clines = ax_comp.add_collection(clc)
        xs = comps[i][:,0]
        ys = comps[i][:,1]
        zs = comps[i][:,2]
        c = np.arange(comps.shape[1])
        sax = ax_comp.scatter(xs, ys, zs, c=c, cmap='viridis',alpha=0.6)
        ax_comp.grid(False)
        ax_comp.set_xticks([])
        ax_comp.set_yticks([])
        ax_comp.set_zticks([])
    
    plt.show()
    return fig

    


# plot critical points
def plot_critical(cs, num_samples, samples):
    # TODO: try changing to plotly interactive plots
    
    fig, axs = plt.subplots(num_samples, 2, sharex=True, sharey=True, subplot_kw=dict(projection='3d'))
    
    for i in range(num_samples):
        # plot original input points
        ax_orig = axs[i,0]
        xs = samples[i][:,0]
        ys = samples[i][:,1]
        zs = samples[i][:,2]
        c = np.arange(samples.shape[1])
        orig = ax_orig.scatter(xs, ys, zs, c=c, cmap=plt.get_cmap('viridis'))
        ax_orig.grid(False)
        
        # plot associated critical points
        ax_crit = axs[i,1]
        # remove duplicates before plotting
        cs_min = np.unique(cs[i], axis=0)
        xs = cs_min[:,0]
        ys = cs_min[:,1]
        zs = cs_min[:,2]
        #c = np.arange(cs.shape[1])
        crit = ax_crit.scatter(xs,ys,zs)
        ax_crit.grid(False)
        
    # add plot wide attributes here
    fig.colorbar(orig, ax = axs[:,0], pad=0.2, shrink=0.6)
    
    return fig


# plot upper points
def plot_upper(ups, num_samples, samples):
    # TODO: try changing to plotly interactive plots
    fig = plt.figure()
    axs = plt.subplots(num_samples, 2, sharex=True, sharey=True, subplot_kw=dict(projection='3d'))
    
    for i in num_samples:
        # plot original input points
        ax_orig = axs[i*2]
        xs = samples[i][:,0]
        ys = samples[i][:,1]
        zs = samples[i][:,2]
        ax_orig.scatter(xs, ys, zs)
        
        # plot associated critical points
        ax_crit = axs[(i*2)+1]
        xs = ups[i][:,0]
        ys = ups[i][:,1]
        zs = ups[i][:,2]
        ax_crit.scatter(xs,ys,zs)
        
    # add plot wide attributes here
    
    return fig



# plot an upper bound shape generated by extracting unique(thresholded) points from a set of aligned CS
# TODO: define additional modes for alternate visualisations
def plot_upper_bound(point_array, mode='scatter', colour_mode = 'height', cmap='viridis'):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    
    xs = point_array[:,0]
    ys = point_array[:,1]
    zs = point_array[:,2]
    
    if colour_mode == 'height':
        c = zs
    elif colour_mode == 'distance':
        lengths = np.linalg.norm(point_array,axis=1)
        c = lengths/np.max(lengths)    
    
    if mode == 'scatter':       
        ax.scatter(xs,ys,zs,c=c,cmap=cmap)        
    elif mode == 'scatter shell':        
        # use the distance from 0 to define the alpha of each point(length of the vector described by the point)
        lengths = np.linalg.norm(point_array,axis=1)
        alphas = lengths/np.max(lengths)
        ax.scatter(xs,ys,zs,c=c,cmap=cmap,alpha=alphas)        
    else:
        return 0
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    return fig






# plot mean critical set
def plot_mean_critical():
    # TODO
    return


# plot mean upper set
def plot_mean_upper():
    # TODO
    return


def plot_critical_umap2D(cs, num_samples, samples):
    
    fig, axs = plt.subplots(num_samples, 2, sharex=True, sharey=True)
    
    for i in range(num_samples):
        # plot original input points
        ax_orig = axs[i,0]
        xs = samples[i][:,0]
        ys = samples[i][:,1]
        ax_orig.scatter(xs, ys)
        
        # plot associated critical points
        ax_crit = axs[i,1]
        xs = cs[i][:,0]
        ys = cs[i][:,1]
        ax_crit.scatter(xs,ys)
    
    return fig









# Trajectory plotting using plotly
#import plotly.graph_objects as go
#import scipy.interpolate






