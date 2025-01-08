#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:56:47 2021

@author: proxy_loken


Functions and utilities for data loading for behavioural / neural classifiaction / clustering

"""

import numpy as np
import scipy.io
import pickle
import pandas as pd
from hubdt import iterative_smoothing


# Function for loading behavioural encoding
# Assumes data is aligned to start of recording, toggle flag to trim using start frame
def load_encodings(curr_sess, trim=False):
    
    
    mat = scipy.io.loadmat(curr_sess.behav_mat)
    encoding = mat['behav_encoding_5fps']
    if trim:
        # this trim assume the encoding is at 1/3 the original recorded frame rate
        encoding = encoding[curr_sess.frame_start:,:]
    
    return encoding


# Function for loading tracking data (from DLC or manual annotations)
# Aligns data to start of recording
# INPUTS: dlc flag if using tracking from DeepLabCut
# if manual, check for full data set(many sessions only have the first ~2000 frames and will need to be manually aligned)
# if loading from pickle, function assumes that the data is preprocessed and in a dictionary entry keyed 'tracking'
def load_tracking(curr_sess, three_d = False, dlc=False, full=True, smooth=False, from_pickle=False, feats=[]):
    
    # If loading directly from a pickle file
    if from_pickle:
        tracking_file = curr_sess.tracking_pickle
        tracking_dict = load_pickle(tracking_file)
        tracking = tracking_dict['tracking']
        
        return tracking     
    
    
    # if loading from a dlc h5 format dataframe
    if dlc:
        # TODO: Handle this with a warning in the ui element in future
        if not feats:
            return False
        # Load DLC h5 file
        df = load_dlc_hdf(curr_sess.tracking_h5)
        start_frame = curr_sess.frame_start
        
        # strip scorer from df
        df = dlc_remove_scorer(df)
        # trim df to start        
        df = dlc_trim_to_start(df,start_frame)
        # smooth dlc tracking
        # Disabled until I fix the starting frame is low likelihood issue
        if smooth:
            smooth_df = iterative_smoothing.iter_smooth(df)
        else:
            smooth_df = df
        # Extract Selected Features from the df
        smooth_df = dlc_select_feats(smooth_df, feats)
        # Centre and process if 2 camera 3d, return raw tracking otherwise
        # TODO: Make a 2d version of the centering, make the centering features selectable, make centering optional
        if three_d:
            # Process df and centre
            top_feats,bottom_feats = dlc_extract_feats_3d(smooth_df)        
            # separate the two camera views using the feature lists
            top_df, bot_df = dlc_separate_views(smooth_df, top_feats, bottom_feats)
            threed_df = dlc_2cam_to_3d(top_df, bot_df, top_feats, bottom_feats)
            centre_df = align_to_centre_3d(threed_df)
            # convert to array
            centre_arr = df_to_arr_3d(centre_df)
            # remove 'com' (centre of mass) from array: not used in current pipeline
            centre_arr = centre_arr[:,:-3]
        else:
            centre_arr = df_to_arr_2d(smooth_df)        
        
        return centre_arr
    
    # Two options if loading from a .mat file (from manual/matlab based tracking)
    elif full:
        # This load function is tailored for a specific .mat format
        # Change as necessary
        mat = scipy.io.loadmat(curr_sess.tracking_mat)
        com = mat['COM']
        threed = mat['threeDData']
        com = com.astype(int)
        threed = threed.astype(int)
        threed = threed[:,1:,:]
        dists_3d = threed - com
        dists_3d = abs(dists_3d)
        start_frame = curr_sess.frame_start
        dists_3d = dists_3d[start_frame:]
        dists_flat = dists_3d.reshape(-1,27)
        
        return dists_flat
    else:
        mat = scipy.io.loadmat(curr_sess.tracking_mat)
        com = mat['COM']
        threed = mat['threeDData']
        com = com.astype(int)
        threed = threed.astype(int)
        threed = threed[:,1:,:]
        dists_3d = threed - com
        dists_3d = abs(dists_3d)
        dists_flat = dists_3d.reshape(-1,27)
        
        return dists_flat
        


# Function for loading stmtx data
# Aligns data to start of recording (make sure this isn't double trimming)
# Also trims first unit (should be timestamp)
def load_stmtx(curr_sess, align=True):
    
    
    mat = scipy.io.loadmat(curr_sess.stmtx_mat)
    Stmtx = mat['STbintrim']    
    Stmtx = np.transpose(Stmtx)
    if align:
        st_aligned = Stmtx[curr_sess.start_bin:,1:]
    else:
        st_aligned = Stmtx[:,1:]
    
    return st_aligned


# Function for loading block residuals
# As these are defined on blocks, the data are pre-separated into blocks
# INPUTS: flags for loading behavioural encodings and raw stmtx left as legacy
def load_block_residuals(curr_sess, behav=False, raw=False):
    
    
    mat = scipy.io.loadmat(curr_sess.block_resid_mat)
    
    
    st_food_res = mat['st_food_res']
    st_shock_res = mat['st_shock_res']
    st_neut_res = mat['st_neut_res']    
    
    data = [st_neut_res, st_food_res, st_shock_res]
    
    # if necessary, behavioural encodings can be loaded here
    if behav:
        beh_food = mat['beh_food']
        beh_shock = mat['beh_shock']
        beh_neut = mat['beh_neut']
        data.append(beh_neut)
        data.append(beh_food)
        data.append(beh_shock)
            
    # if necessary, raw st_bins casn be loaded here
    if raw:
        st_food = mat['st_food']    
        st_shock = mat['st_shock']
        st_neut = mat['st_neut']
        data.append(st_neut)
        data.append(st_food)
        data.append(st_shock)
            
    return data




# Function for extracting context points from an array of data
# INPUTS: curr_sess - session object for current session, data - array of points (samples x dims)
def extract_context_points(curr_sess, data):
    
    neut_points = data[curr_sess.neutral_start:curr_sess.neutral_end,:]
    food_points = data[curr_sess.food_start:curr_sess.food_end,:]
    shock_points = data[curr_sess.shock_start:curr_sess.shock_end,:]
    
    return neut_points, food_points, shock_points


# Function for creating a label vector for context blocks
# INPUTS: curr_sess - session object for current session, data - array of points (samples x dims)
def create_context_labels(curr_sess, data):
    
    context_labels = np.zeros(np.size(data,0))
    context_array = np.zeros((np.size(data,0),3))
    
    context_labels[:] = -1
    
    context_labels[curr_sess.neutral_start:curr_sess.neutral_end]= 0
    context_array[curr_sess.neutral_start:curr_sess.neutral_end,0]=1
    context_labels[curr_sess.food_start:curr_sess.food_end]= 1
    context_array[curr_sess.food_start:curr_sess.food_end,1]=1
    context_labels[curr_sess.shock_start:curr_sess.shock_end]= 2
    context_array[curr_sess.shock_start:curr_sess.shock_end,2]=1
    
    return context_labels,context_array



# Pickling/Unpickling Data


def save_pickle(data, filename):
    
    pickle.dump(data, open(filename, 'wb'))
    


def load_pickle(filename):
    
    with open(filename,'rb') as fp:
        l_data = pickle.load(fp)
        
    return l_data





# Trimmin / slicing functions

# Function to trim pre-period from data set
# Assumes the set is alinged to start of recording already
def trim_pre(curr_sess, data):
    
    
    starts = [curr_sess.food_start, curr_sess.shock_start, curr_sess.neutral_start]
    
    pre = np.min(starts)
    
    trimmed_data = data[pre:,:]
    
    return trimmed_data


# Function for trimming data sets to match in length (by removing excess from the end)
# Assume both sets are aligned to start of recording already
def trim_to_match(data1,data2):
    
    if len(data1) > len(data2):
        data1 = data1[0:len(data2)]
    else:
        data2 = data2[0:len(data1)]
    
    return data1, data2


# Function to extract blocks from data set
# Assumes the set is aligned to start of recording already
def extract_blocks(curr_sess, data):
    
    
    data_neutral = data[curr_sess.neutral_start:curr_sess.neutral_end+1,:]
    data_food = data[curr_sess.food_start:curr_sess.food_end+1,:]
    data_shock = data[curr_sess.shock_start:curr_sess.shock_end+1,:]
        
    return data_neutral, data_food, data_shock



# Utlities (Augmentation, categories, etc.)


# Function for creating a new label set with an extra category labels for any unlabelled frames
# INPUTS: Labels is an array (observations by categories) of binary labels
def add_unlabelled_cat(labels):
    
    inds = np.where(~labels.any(axis=1))[0]
    cols = np.size(labels,axis=0)
    rows = np.size(labels,axis=1)
    add_labels = np.zeros((cols,rows+1))
    add_labels[:,:-1] = labels
    add_labels[inds,-1] = 1
    
    return add_labels


# Helper for downsampling data to a lower rate
def convert_fr(data,in_fr,out_fr):
    
    in_fr = int(in_fr)
    out_fr = int(out_fr)
    step = in_fr//out_fr
    out = data[::step,:].copy()
    
    return out




# DATAFRAME functions


# loader for DLC dataframe
def load_dlc_hdf(filename):
    
    dlc_df = pd.read_hdf(filename)
    
    return dlc_df


# Function for slicing DLC hdf dataframes for specific features
# ONLY WORKS ON DLC DATAFRAMES (the slicing is dependent on the multindex arrangement)
# Strips the "scorer" index (irrelevent in this analysis) 
def select_dlc_feats(data, feats):
    
    idx = pd.IndexSlice
    out = data.loc[:,idx[:,feats,:]].copy()    
    out = out.droplevel('scorer',axis=1)
    return out


#Helper to strip the "scorer" index from a DLC dataframe
#If you have multiple scorers on a single dataset you may want to retain this index
def dlc_remove_scorer(data):
    
    return data.droplevel('scorer',axis=1)


# Helper to trim the beginning of a DLC df
# If using starting bin, ensure to multiply/divide to match the frame rate
def dlc_trim_to_start(data, start_frame):
    
    idx = pd.IndexSlice
    #out = data.loc[:,idx[:,start_frame:]].copy()
    out = data.iloc[start_frame:,:].copy()
    
    return out



# Function to extract feature names from a 3d(2 camera) DLC dataset and divide the features list by camera
# Dependent on our particular camera arrangement
def dlc_extract_feats_3d(data):
    
    all_feats = list(data.T.index.get_level_values(0).unique())
    top_feats = all_feats[:len(all_feats)//2]
    bottom_feats = all_feats[len(all_feats)//2:]
    
    return top_feats,bottom_feats


# Function to get features for selection
def dlc_get_feats(curr_sess):
    data = load_dlc_hdf(curr_sess.tracking_h5)
    data = dlc_remove_scorer(data)
    all_feats = list(data.T.index.get_level_values(0).unique())
    
    return all_feats


# Function to separate the two camera views from a 2 view DLC dataframe
def dlc_separate_views(data, top_feats, bottom_feats):
    
    idx = pd.IndexSlice
    top_df = data.loc[:,idx[top_feats,:]].copy()
    bot_df = data.loc[:,idx[bottom_feats,:]].copy()    
    
    return top_df, bot_df

# Helper to pull specific features from a DLC dataframe
# TODO: combine with slightly different version above, retain functionality
def dlc_select_feats(data, feats):
    
    idx = pd.IndexSlice
    out_df = data.loc[:,idx[feats,:]].copy()
    
    return out_df


# Function to combine the two single view DLC dataframes into one
# Takes the 'y' value from view 1(top) as the 'z' value for view 2
# Returns augmented view 2
def dlc_2cam_to_3d(top_df, bot_df, top_feats, bottom_feats):
    
    data = bot_df.copy()
    
    for i in range(len(bottom_feats)):
        
        data.loc(axis=1)[bottom_feats[i],'z'] = top_df.loc(axis=1)[top_feats[i],'y']
    
    
    return data




# Function to align to generated centre of mass for a 3d DLC dataset
# TEMP: requires a specific set of features
# TODO: Take in feature from laoding select, also make centering optional
def align_to_centre_3d(df):
    
    data = df.copy()
    
    cen_x = (data.loc(axis=1)['shoulder2','x'].to_numpy() + data.loc(axis=1)['pelvis2','x'].to_numpy()) / 2
    
    cen_y = (data.loc(axis=1)['shoulder2','y'].to_numpy() + data.loc(axis=1)['pelvis2','y'].to_numpy()) / 2
    
    cen_z = (data.loc(axis=1)['shoulder2','z'].to_numpy() + data.loc(axis=1)['pelvis2','z'].to_numpy()) / 2
    
    for feature in list(data.T.index.get_level_values(0).unique()):
        
        data.loc(axis=1)[feature,'x'] = data.loc(axis=1)[feature,'x']-cen_x
        data.loc(axis=1)[feature,'y'] = data.loc(axis=1)[feature,'y']-cen_y
        data.loc(axis=1)[feature,'z'] = data.loc(axis=1)[feature,'z']-cen_z
        
    data.loc(axis=1)['com','x'] = cen_x
    data.loc(axis=1)['com','y'] = cen_y
    data.loc(axis=1)['com','z'] = cen_z
    
    return data


# Function to pull x,y,z points from dataframe and turn into numpy array
def df_to_arr_3d(data):
    
    darray = []
    
    for feature in list(data.T.index.get_level_values(0).unique()):
        darray.append(data.loc(axis=1)[feature,['x','y','z']].to_numpy())
    
    out = np.concatenate(darray, axis=1)
    
    return out

# as above but for 2-d tracking
def df_to_arr_2d(data):
    
    darray= []
    
    for feature in list(data.T.index.get_level_values(0).unique()):
        darray.append(data.loc(axis=1)[feature,['x','y']].to_numpy())
    
    out = np.concatenate(darray, axis=1)
    
    return out

# Helper to call interative smoothing tracking thresholding
def threshold_tracking(data, threshold):
    
    return iterative_smoothing.threshold_tracking_np(data, threshold)


    
