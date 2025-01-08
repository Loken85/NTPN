# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:48:26 2019

@author: proxy_loken

Utilities for automated behavioural state tracking with deeplabcut input
"""

import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.stats import sem
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder


# Turns a dataframe into a dictionary of numpy arrays
# one (length X 3) array for each part
def frame_to_darray(data):
    darray = {}
    
    for feature in list(data.T.index.get_level_values(0)):
        darray[feature] = data[feature].to_numpy()        
    
    
    return darray

# saves a .mat file for use in matlab. Takes in a dataframe, saves as a 
# struct containing a set of numpy arrays
# Currently saves to PWD, TODO: set the filepath dynamically
def expdf_to_matlab(data):
    arry = frame_to_darray(data)
    
    sio.savemat('data_mat.mat', {'data': arry})
    

# same as above, but takes in dict or numpy array
# could be combined with a type query...but whatever
def expnp_to_matlab(data):
    
    sio.savemat('data_mat.mat', {'data': data})
    
    
# create a raw projections dataframe. extracts just x and y positions for features
# in feats list
def create_raw_projections(data,feats):
    
    data = data.T
    data = data.sort_index()
    rproj = data.loc[(feats,('x','y')),:]
            
    rproj = rproj.T    
    return rproj

# aligns the projections to a selected feature. Selected feature positions will
# be all zeros 
def align_projections2D(rproj,a_ft,dim):
    aproj = rproj.copy()
    # assemble the full feature lable
    feature = a_ft + str(dim)
    aproj = aproj.subtract(aproj[feature], level=1)
    
    return aproj

# redundant
def align_projections3D(rproj,a_ft,n_ft):
    
    return aproj


# plots the wave aplitudes of a single feature projected into wavelet space
# version for coming from dataframe
# TODO: add figure name and axes labels
def plot_wamps(scalo,freqs,name, figs_per_row=5):
    # number of subplots
    num_plts = np.size(scalo,0)
    # number of rows 
    n_rows = np.ceil(num_plts/figs_per_row)
    n_rows = n_rows.astype(int)
    # create suplots. set x and y axes to be shared, and set spacing
    #fig, axs = plt.subplots(n_rows,figs_per_row,sharex=True,sharey=True,gridspec_kw={'hspace': 0,'wspace' : 0})
    # version without shared axis
    fig, axs = plt.subplots(n_rows,figs_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig.suptitle(name)    
    # only use outer axes labels
    for ax in axs.flat:
        ax.label_outer()
    
    for r in range(0,n_rows):
        for n in range(0,figs_per_row):
            curr = r*figs_per_row + n
            if curr<num_plts:
                axs[r,n].plot(scalo[curr,:])
            else:
                break
    

# plots the wave aplitudes of a single feature projected into wavelet space
# version for coming from dataframe
# TODO: add figure name and axes labels
def plot_wamps_np(projections,freqs,name, colour='green', figs_per_row=5, hide_axis=True):
    projections = np.transpose(projections)
    # number of subplots
    num_plts = np.size(projections,0)
    # number of rows 
    n_rows = np.ceil(num_plts/figs_per_row)
    n_rows = n_rows.astype(int)
    # create suplots. set x and y axes to be shared, and set spacing
    fig, axs = plt.subplots(n_rows,figs_per_row,sharex=True,sharey=True,gridspec_kw={'hspace': 0,'wspace' : 0})
    # version without shared axis
    #fig, axs = plt.subplots(n_rows,figs_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig.suptitle(name,fontsize=20)    
    # only use outer axes labels
    for ax in axs.flat:
        ax.label_outer()
    
    for r in range(0,n_rows):
        for n in range(0,figs_per_row):
            curr = r*figs_per_row + n
            if curr<num_plts:
                axs[r,n].plot(projections[curr,:],color=colour)
                if hide_axis:
                    axs[r,n].tick_params(which='both',left=False,right=False,labelleft=False,top=False,bottom=False,labelbottom=False)
            else:
                break



    
# generates and plots a graphical object showing the current frame location in the
# clustered reduced dimensionality space
def plot_curr_cluster(t_out, labels, frame, xi, yi):
    
    # create figure
    fig, ax = plt.subplots()
    # plot the clusters
    plt.pcolormesh(xi, yi, labels)
    # plot location of current frame (x, y reversed because t_out is transposed)
    y, x = t_out[frame]
    plt.scatter(x, y, s=10, c='red', marker='+')
    
    plt.show()
    


# generates and saves a movie of the t-sne space locations for each frame
def cluster_anim(t_out, labels, xi, yi, fps, start_f = 0, end_f = 1000):
    metadata = dict(title="T-sne Space Plot", artist="matplotlib", comment="Movie of t-sne space locations for each frame")
    
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    
    fig, ax = plt.subplots()
    
    #plt.xlim(np.min(xi)-5,np.max(xi)+5)
    #plt.ylim(np.min(yi)-5,np.max(yi)+5)
    
    #frames = np.size(t_out, 0)
    frames = end_f-start_f
    
    with writer.saving(fig, "location_plot.mp4", frames):
        for i in range(start_f,end_f):
            plt.pcolormesh(xi, yi, labels)
            
            ax.autoscale(False)
            
            y, x = t_out[i]
            plt.scatter(x,y,s=10, c='red', marker='+')
            
            writer.grab_frame()
            
            
# helper function for finding the index of the nearest value in an array
# note: this will be slow on large arrays
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# function to build and add z values(from density) to points to enable 3d plotting
# INPUTS: points - array(2d) of points you want the fetch z values for
# xi,yi - grids for x and y, density - z grid (density of tsne space, usually)
# OUTPUTS: 3d array of points with z values
def create_zmap(points, xi, yi, density):
    
    num_points = np.size(points,0)
    out_points = np.zeros((num_points, 3))
    out_points[:,:-1] = points
    
    for i in range(num_points):
        idx = find_nearest_index(xi[:,0], points[i,0])
        idy = find_nearest_index(yi[0,:], points[i,1])
        out_points[i,2] = density[idx,idy]
        
    return out_points


def plot_ethogram(l_frames):
    #max_l = np.max(l_frames)
    frames = np.size(l_frames,0)
    
    #ys = range(0,max_l)
    xs = range(0, frames)
    
    fig,ax = plt.subplots()
    
    plt.scatter(xs, l_frames, c=l_frames, s=10, cmap='jet')
    
    plt.show()


def plot_ethograms(l_frames):
    #max_l = np.max(l_frames)
    frames = np.size(l_frames,0)
    
    #ys = range(0,max_l)
    xs = range(0, frames)
    
    n_rows = np.size(l_frames,1)
    
    fig, axs = plt.subplots(n_rows,1,gridspec_kw={'hspace': 0,'wspace' : 0})
    
    for ax in axs.flat:
        ax.label_outer()
        ax.set_ylim([-0.5,2.5])
    
    for i in range(0,np.size(l_frames,1)):
        
        axs[i].scatter(xs, l_frames[:,i], c=l_frames[:,i], s=10, cmap='jet')
    
    plt.show()

    
# count number of occurances of labels in label array 
def count_labels(l_array):
    labels, counts = np.unique(l_array, return_counts=True)
    
    return labels, counts

# count consecutive occurances of labels in label vector
# returns a list of counts for each label, and a list of the index with the biggest count
def count_consecutive_labels(l_array):
    l_array = l_array.astype(int)
    counts_list = []
    max_inds_list = []
    
    for i in range(0,(np.max(l_array)+1)):
        
        bool_arr = l_array==i
        start_end_inds = np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0]
        diffs = np.diff(start_end_inds)[::2]
        max_count_ind = start_end_inds[(np.argmax(diffs)*2)]
        counts_list.append(diffs)
        max_inds_list.append(max_count_ind)
        
    return counts_list,max_inds_list

# 2d array version
# count consecutive occurances of labels in label array
# returns a list of counts for each label, and a list of the index with the biggest count
def count_consecutive_labels2(labels):
    counts_list = []
    max_inds_list = []
    
    for i in range(0,np.size(labels,1)):
        
        bool_arr = labels[:,i] == 1
        start_end_inds = np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0]
        diffs = np.diff(start_end_inds)[::2]
        max_count_ind = start_end_inds[(np.argmax(diffs)*2)]
        counts_list.append(diffs)
        max_inds_list.append(max_count_ind)
        
    return counts_list,max_inds_list

# Helper for filling gaps in consecutive labels (brief sets of noise labels within a longer label period)
# instances of noise less than the gap size and flanked by a consistent label will be replaced with that label
def gap_fill_labels(labels, noise_label=-1, gap_size=5):
    
    out_labels = np.copy(labels)
    
    bool_arr = labels == noise_label
    start_end_inds = np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0]
    if np.size(start_end_inds) % 2:
        start_end_inds = start_end_inds[:-1]
    diffs = np.diff(start_end_inds)[::2]
    diffs_bool = diffs <= gap_size
    start_end_mask = diffs_bool.repeat(2)
    start_end_trim = start_end_inds[start_end_mask]
    
    for i in range(np.sum(diffs_bool)):
        
        start_ind = start_end_trim[i*2]
        end_ind = start_end_trim[i*2+1]
        
        if start_ind <=0:
            continue
        
        pre_label = labels[start_ind-1]
        post_label = labels[end_ind]
        
        if pre_label==post_label:
            out_labels[start_ind:end_ind] = pre_label
            
    
    return out_labels


# Helper for removing label jitter (defined here as instances of conecutive laebl counts less than the jitter size)
# in the output, short label instances will be replaced with noise labels
def remove_label_jitter(labels, noise_label=-1, jitter_size=3):
    
    out_labels = np.copy(labels)
    
    for i in range(0,(np.max(labels)+1)):
        
        bool_arr = labels==i
        start_end_inds = np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0]
        if np.size(start_end_inds) % 2:
            start_end_inds = start_end_inds[:-1]
        diffs = np.diff(start_end_inds)[::2]
        diffs_bool = diffs < jitter_size
        start_end_mask = diffs_bool.repeat(2)
        start_end_trim = start_end_inds[start_end_mask]
        
        for j in range(np.sum(diffs_bool)):
            
            out_labels[start_end_trim[j*2]:start_end_trim[j*2+1]] = noise_label
            
    
    return out_labels
    



# function to take the mean of consecutive labels for each category
def mean_consecutive_labels(counts_list):
    means_list = []
    
    for i in range(0,len(counts_list)):
        
        cat = counts_list[i]
        cat_num = np.sum(cat)
        cat_curr = 0
        
        for j in range(0,np.size(cat,0)):
            cat_curr = cat_curr + (cat[j]*(j+1))
        
        cat_mean = cat_curr/cat_num
        means_list.append(cat_mean)        
    
    return means_list



    
# trim consecutive count arrays to discard short "behaviours"
def trim_counts(counts_list,threshold=5):
    
    trim_counts = []
    
    for count in counts_list:
        
        count = np.sort(count)
        
        inds = np.where(count>=threshold)
        trim = count[inds]
        
        trim_counts.append(trim)
        
    return trim_counts


# plot a set of counts as histograms
def plot_label_counts(counts_list, plots_per_row=3, name='Label Counts',color='blue'):
    
    max_count = 0
    
    for count in counts_list:
        if count.any():
            curr_max = np.max(count)
        if curr_max > max_count:
            max_count = curr_max
    
    num_plots = len(counts_list)
    n_rows = np.ceil(num_plots/plots_per_row)
    n_rows = n_rows.astype(int)
    
    bins = range(0,max_count+1)
    
    
    fig, axs = plt.subplots(n_rows,plots_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig.suptitle(name)    
    # only use outer axes labels
    for ax in axs.flat:
        ax.label_outer()
    
    for r in range(0,n_rows):
        for n in range(0,plots_per_row):
            curr = r*plots_per_row + n
            if curr<num_plots:
                axs[r,n].hist(counts_list[curr],bins=bins,color=color)
            else:
                break


# converts probabilistic predictions to labels by thresholding
def threshold_labels(preds, threshold = 0.05):
    labels = np.zeros((np.size(preds,0),np.size(preds,1)))
    inds = preds > threshold
    labels[inds] = 1
    
    return labels


# smooths labels via convolution and then rounds back to a binary array
def smooth_labels(data, span):
    box = np.ones(span)/span
    smooth_data = np.convolve(data,box,mode='same')
    
    round_data = np.round(smooth_data,decimals=0)
    
    return round_data.astype(int)


# smooth a set of cluster labels over a set span
# expects a vector of labels(samplesxNone)
def smooth_clusters(data, span):
    data = data.reshape(-1,1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data)
    onehot = enc.transform(data).toarray()
    for i in range(np.size(onehot,1)):
        onehot[:,i] = smooth_labels(onehot[:,i],span)
    
    out = enc.inverse_transform(onehot)
    # if no category is assigned, inverse transform puts in None. Replace with -1 (Noise label)
    out[out==None] = -1
    return np.squeeze(out.astype(int))


# helper to convert label indices to actual frame counts
# INPUTS: inds - vector of indices to convert, offset - int offset from start of video
# NOTE: this assumes video recorded at 15fps, and labels etc. done at 5 fps
# default offset is zero, (for 0330i231 video for exmaple, offset is 379 if not already taken off)
# TODO: change to take in video params(might need a re-format to make sense)
def indices_to_frames(inds, sfreq, vid_fr, offset = 0):
    
    # deprecated
    # if there is a difference between the label 'framerate' and th video framerate, calculate the adjustment
    adjustment = vid_fr/sfreq
    frames = np.copy(inds)
    frames = frames*adjustment
    frames = frames + offset
    
    return frames


# single frame version of above helper
def index_to_frame(ind, sfreq, vid_fr, offset = 0):
    
    # deprecated
    # if there is a difference between the label 'framerate' and th video framerate, calculate the adjustment
    adjustment = vid_fr/sfreq
    frame = np.copy(ind)
    frame = frame*adjustment
    frame = int(frame + offset)
    
    return frame


# helper to generate frame indices for use in exrtacting snippets from video
# INPUTS: start_index - start of the behaviour of interest, length - length in seconds of frames to gen
# fps - fps of the video
# NOTE: this assumes that the labels etc. need to be converted to match the video,
# as such it uses the indices_to_frames function to do this conversion
def gen_frames_for_gif(start_index,length=2,fps=15):
    
    start_frame = index_to_frame(start_index)
    frames = list(range(start_frame,(start_frame+(fps*length)),1))
    
    return frames
    
    



# calculates the corrcoef between two sets of variables of the same length
# note: this is rowwise corrcoef, so make sure the inputs arrays are var*obs
def corr2_coef(A, B):
    # Rowwise mean of input arrays & then subtract from imput arrays
    A_mA = A-A.mean(1)[:,None]
    B_mB = B-B.mean(1)[:,None]
    
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    
    # calc corr then change to coef by normalisation
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:,None],ssB[None]))




# calculates the average per neuron firing rate differences between two matched sets of binned FR
# notes:  the two sets must be the same length and aligned, the measure is non-symmetric
def calc_avg_fr_diff(stbin1,stbin2):
    
    st1_deltas = np.zeros(np.size(stbin1,1))
    st2_deltas = np.zeros(np.size(stbin2,1))
    
    
    for i in range(np.size(stbin1,1)):
        avg_st1 = np.mean(stbin1[:,i])
        avg_st2 = np.mean(stbin2[:,i])
        
        diff = stbin1[:,i]-stbin2[:,i]
        diff_avg = np.mean(diff)
        
        delta_st1 = diff_avg/avg_st1
        delta_st2 = diff_avg/avg_st2
        
        st1_deltas[i] = delta_st1
        st2_deltas[i] = delta_st2
        
    return st1_deltas, st2_deltas
    

# Function for transforming a matrix of differences(from cal avg fr diff) into proportions of the mean of the raw value
# Used when preparing raw v residual differences for plotting
def diff_to_mean_proj(raw_values, diffs):
    
    raw_means = np.mean(raw_values,0)
    
    prop_del = diffs - 1
    
    mean_projs = raw_means*prop_del
    
    return mean_projs




    
# plots average per neuron firing rate differences as bars
def plot_avg_fr_diffs(fr_deltas):
    
    fig, axs = plt.subplots()
    
    axs.bar(np.arange(np.size(fr_deltas)),fr_deltas)
    
    fig.suptitle("Average per Neuron Firing Rate Deltas")
    
    return fig
    


def calc_resid_diffs(targets, residuals):
    
        
    diffs = targets-residuals
    
    mean_diffs = np.mean(diffs,0)
    
    return abs(mean_diffs)


def plot_resid_diffs(resid_diffs, colour='cornflowerblue'):
    
    fig, ax = plt.subplots()
    
    bar_locs = np.arange(np.size(resid_diffs))
    
    ax.bar(bar_locs, resid_diffs, color=colour)
    
    return fig


# Function for calculating the amount of firing attributed to a target via r2 values
# Returns FR values (as a proportion of mean firing) (1 per neuron)
def calc_fr_props(raw_data, r2s):
          
    mean_frs = np.mean(raw_data, 0)
    
    fr_props = mean_frs * r2s    
    
    return fr_props


# Helper to plot firing rate proportions
def plot_fr_props(fr_props, colour='cornflowerblue', prod=False):
    
    fig, ax = plt.subplots()
    
    bar_locs = np.arange(np.size(fr_props))
    
    ax.bar(bar_locs, fr_props, color=colour)
    
    if prod:
        
        ax.set_ylim([0,0.2])
        #ax.set_xlim([-5,180])
        
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    return fig




# gets the indices of a particular cluster from the labels array
def get_cluster_indices(labels, cluster):
    inds = np.argwhere(labels[:,cluster] >= 1)
    return inds



# gets the indices of true labels that are also present in the predicted label
# set. Returns both the index set from the true labels as well as a boolean
# array matching true labels showing predicted label matches
def get_comparative_labels(true_labels, pred_labels,cat):
    # fetch indices for each set of labels
    true_indices = get_cluster_indices(true_labels,cat)
    pred_indices = get_cluster_indices(pred_labels,cat)
    
    comparative_bool = np.isin(true_indices[:,0], pred_indices[:,0])
    comparative_indices = pred_indices[comparative_bool,0]
    
    return comparative_indices, comparative_bool
    
    

def create_cluster_image_indices(poses, start_frame = 379, offset = 0):
    # create an array of actual frame indices for each pose 
    frame_indices = np.arange(0,np.size(poses,0),1)
    frame_indices = frame_indices*3 + (start_frame) + offset
    return frame_indices


def get_cluster_images(frame_indices, labels, category):
    # retrieves indices for frames from a given cluster/category
    inds = np.argwhere(labels[:,category])
    cluster_inds = frame_indices[inds]
    return cluster_inds


    
    

def plot_confuse(t_labels, p_labels):
    
    # define plotting layout
    
    n_labs = np.size(t_labels,1)
    n_rows = n_labs//3 + n_labs%3
    n_cols = 3
    
    fig, axs = plt.subplots(n_rows,n_cols)
    axs = axs.ravel()
    
    for i in range(np.size(t_labels,1)):
        disp = ConfusionMatrixDisplay(confusion_matrix(t_labels[:,i],p_labels[:,i]), display_labels=[0,i])
        disp.plot(ax=axs[i], values_format='.4g')
        disp.ax_.set_title(f'class{i}')
        if i<(n_rows*n_cols-n_cols):
            disp.ax_.set_xlabel('')
        if i%n_cols!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    
    plt.subplots_adjust(wspace=0.10, hspace=0.10)
    fig.colorbar(disp.im_, ax=axs)
    plt.show()
    
    
    return 0




# cumulative counting of 1s in a given category. Makes a matching length array with 1s
# replaced with counts of consecutive 1s in the current run at that index
def cumulative_cat_labels(labels, category):
    label = labels[:,category]
    out = np.concatenate([np.cumsum(count) if count[0]==1 else count for count in np.split(label, 1+np.where(np.diff(label))[0]) ])
    
    return out




# function to convert projections array to array of combined x,y magnitude (2D VERSION)
# INPUTS: data - array of wavelet amplitudes (samples by x * num_waves + y * num_waves)
# num_waves - number of wavelets used to create projections array
def convert_wav_amps_2d(data, num_waves):
    
    num_feats = (np.size(data,1)//num_waves)//2
    
    out = np.zeros((np.size(data,0), np.size(data,1)//2))
    
    for i in range(num_feats):
        
        for j in range (num_waves):
        
            x_ind = i*num_waves*2 + j
            y_ind = i*num_waves*2 + num_waves + j
            out[:,i*num_waves+j] = (data[:,x_ind] + data[:,y_ind])/2
    
    return out





# function to correct for wavelength in projections data
# multiplies projections by wavelength
# INPUTS: data - array of wavelet magnitudes (samples by x(y) * num_waves)
# freqs - frequencies of the wavelets
def correct_for_wavelength(data, freqs):
    
    out = np.zeros(np.shape(data))
    # get number of features(x + y)
    num_freqs = len(freqs)
    num_feats = np.size(data,1)//num_freqs
    
    for i in range(num_feats):
        
        for j in range(num_freqs):
            
            out[:,(i*num_freqs + j)] = data[:,(i*num_freqs) + j] * freqs[j]
            
    return out


# function to correct for average wave response in projections data
# divides by average response for a given wavelet
# INPUTS: data - array of wavelet magnitudes (samples by x(y) * num_waves)
# freqs - frequencies of the wavelets
def correct_for_response(data, freqs):
    
    out = np.zeros(np.shape(data))
    num_freqs = len(freqs)
    num_feats = np.size(data,1)//num_freqs
    
        
    for i in range(num_freqs):
        
        indices = range(i,(np.size(out,1)-(num_freqs-i)+1),num_freqs)
        
        response_sum = np.sum(data[:,indices])
        response_mean = response_sum/(np.size(data,0)*num_feats)
        
        out[:,indices] = data[:,indices]/response_mean
        
    return out
            
            
# function to average across dimensions of wavelet projection (default x and y)
# takes mean of x and y wavelet response
# INPUTS: data - array of wavelet magnitudes (samples by x(y) * num_waves)
# freqs - frequencies of the wavelets, dims - dimension of projections(default 2)
def calc_mean_response(data, freqs, dims=2):
    
    data_shape = np.shape(data)
    out = np.zeros((data_shape[0],data_shape[1]//dims))
    num_freqs = len(freqs)
    #num_feats = np.size(data,1)//num_freqs
    #num_feats_out = num_feats//dims
    
    for i in range(num_freqs):
        
        if dims == 2:
            
            x_indices = range(i,(np.size(data,1)-(num_freqs-i)+1),num_freqs*dims)
            y_indices = range(i+num_freqs,(np.size(data,1)-(num_freqs-i)+1),num_freqs*dims)
                        
            x_mags = data[:,x_indices]
            y_mags = data[:,y_indices]
                        
            mean_mags = (x_mags + y_mags) / dims
            #mean_mags = np.mean(np.array([x_mags,y_mags]), axis=0)
            
            out_inds = range(i,(np.size(out,1)),num_freqs)
            out[:,out_inds] = mean_mags
            
        elif dims == 3:
            
            x_indices = range(i,(np.size(data,1)-(num_freqs-i)),num_freqs*dims)
            y_indices = range(i+num_freqs,(np.size(data,1)-(num_freqs-i)),num_freqs*dims)
            z_indices = range(i+num_freqs*2,(np.size(data,1)-(num_freqs-i)+1),num_freqs*dims)
            
            x_mags = data[:,x_indices]
            y_mags = data[:,y_indices]
            z_mags = data[:,z_indices]
            
                        
            mean_mags = (x_mags + y_mags + z_mags) / dims
            #mean_mags = np.mean(np.array([x_mags, y_mags, z_mags]), axis=0)
            out_inds = range(i,(np.size(out,1)),num_freqs)
            out[:,out_inds] = mean_mags
            
        else:
            break
    
        
    return out




# function to plot wavelet magnitudes for a given single cluster labelling
# INPUTS: data - array of wavelet magnitudes (or amplitudes, either will work) (samples by x(y) * num_waves)
# labels - label vector for samples
# cluster - cluster label to select
# features - list of features(split for x and y if 2d, x y and z if 3d, not split if averaging dims)
# freqs - frequencies of the wavelets
def plot_cluster_wav_mags(data, labels, cluster, feats, freqs, dims=2, wave_correct = False, response_correct = False, mean_response = False, colour = 'blue'):
    
    
    # plot params
    plots_per_row = 2
    num_waves = len(freqs)
    if not feats:
        num_feats = np.size(data,1)/num_waves
        feats = list(np.arange(0,num_feats))
    else:
        num_feats = len(feats)
    num_plots = num_feats
    bar_locs = np.arange(num_waves)
    n_rows = np.ceil(num_plots/plots_per_row)
    n_rows = n_rows.astype(int)
    
    #fig, axs = plt.subplots(n_rows,plots_per_row,sharey= True, gridspec_kw={'hspace': 0.2,'wspace' : 0.2})
    fig, axs = plt.subplots(n_rows,plots_per_row,sharey= True,sharex=True,figsize=(5,6))
    
    
    if wave_correct:
        data = correct_for_wavelength(data, freqs)
        
    if response_correct:
        data = correct_for_response(data, freqs)
        
    if mean_response:
        data = calc_mean_response(data, freqs, dims=dims)
    
    
    clust_data = data[(labels==cluster),:]
    
    ax = axs.ravel()
    
    tot_max = 0
    
    for i, feat in enumerate(feats):
        
        means = np.mean(clust_data[:,(i*num_waves):(i*num_waves+num_waves)],axis=0)
        errs = sem(clust_data[:,(i*num_waves):(i*num_waves+num_waves)],axis=0)
        curr_max = max(means,default=0)
        if not errs.any():
            errs = np.zeros((num_waves))
        if not means.any():
            means = np.zeros((num_waves))
        if curr_max > tot_max:
            tot_max = curr_max
        
        ax[i].bar(bar_locs, means, yerr=errs, error_kw=dict(lw=3,capsize=3,capthick=2), align='center', color=colour)
        #ax[i].set_ylabel('MWA')
        ax[i].set_xticks(bar_locs)
        ax[i].set_xticklabels(np.round(freqs, 1))
        ax[i].set_title(feat, fontsize=20)
        
    for a in ax:
        a.label_outer()
        a.set_ylim([0,tot_max+0.2*tot_max])
        #a.set_ylim([0,2])
        #a.set_yticklabels([])
    
    # turn off empty plot if necessary
    if num_plots < (n_rows*plots_per_row):
        ax[-1].set_axis_off()
    
    plt.tight_layout()
    return fig
    


# function to plot selected wavelet amplitudes 
# INPUTS: data - array of wavelet magnitudes (or amplitudes, either will work) (samples by x(y) * num_waves)
# wavelets - list of wavelets of interest (from full set of wavelets)
# labels - label vector for samples
# cluster - cluster label to select
# freqs - frequencies of the wavelets
def plot_select_wavelets(data, wavelets, labels, cluster, freqs, wave_correct=False, response_correct=False, colour='blue'):
    
    if wave_correct:
        data = correct_for_wavelength(data, freqs)
        
    if response_correct:
        data = correct_for_response(data, freqs)
    
    selected_wavs = data[:,wavelets]
    
    clust_data = selected_wavs[(labels==cluster),:]
    
    #plot params
    num_waves = len(wavelets)
    bar_locs = np.arange(num_waves)
    
    fig, ax = plt.subplots()
    
    means = np.mean(clust_data,axis=0)
    #errs = np.std(clust_data,axis=0)
    tot_max = np.max(means)
    
    ax.bar(bar_locs, means, align='center', color=colour)
    ax.set_ylabel('MWA')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels([])
    ax.set_ylim([0,tot_max+0.2*tot_max])
    
    plt.tight_layout()    
    
    return 0


def plot_pca_clusters(pca_data, sig_clusters, labels):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    #ax.scatter(pca_data[:,0],pca_data[:,1],pca_data[:,2],s=2,c='grey',alpha=0.05)
    
    points_neut = pca_data[labels == sig_clusters[0]]
    points_food = pca_data[labels == sig_clusters[1]]
    
    ax.scatter(points_neut[:,0],points_neut[:,1],points_neut[:,2],s=5,c='blue')
    ax.scatter(points_food[:,0],points_food[:,1],points_food[:,2],s=10,c='green')



def plot_pca_clusters_overlay(pca_data,sig_clusters, neut_points, food_points, labels):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    #ax.scatter(pca_data[:,0],pca_data[:,1],pca_data[:,2],s=2,c='grey',alpha=0.05)
    
    points_neut = pca_data[labels == sig_clusters[0]]
    points_food = pca_data[labels == sig_clusters[1]]
        
    ax.scatter(points_neut[:,0],points_neut[:,1],points_neut[:,2],s=5,c='blue')
    ax.scatter(points_food[:,0],points_food[:,1],points_food[:,2],s=10,c='green')
    
    ax.scatter(neut_points[:,0],neut_points[:,1],neut_points[:,2],s=5,c='black')
    ax.scatter(food_points[:,0],food_points[:,1],food_points[:,2],s=5,c='grey')




# function to plot pure clusters projected along significant wavelet dimensions
# significant dims are chosen from impact value on SHAP test, clusters from purity measure vs context
# INPUTS: data - full wavelet projections, sig_features - list of 3 features to use as plot axis
# sig_clusters - list of three clusters from label set to plot (one for each context)
# labels - vector of cluster labels 
def plot_sig_projs(data, sig_features, sig_clusters, labels, plot_all=False, norm=False):
    
    sig_projs = data[:,sig_features]
    
    if norm:
        #TODO: standardize data
        print('Not yet implemented')
    
    points_neut = sig_projs[labels == sig_clusters[0]]
    points_food = sig_projs[labels == sig_clusters[1]]
    points_shock = sig_projs[labels == sig_clusters[2]]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # TODO: fix axis labels and ranges
    
    if plot_all:
        ax.scatter(sig_projs[:,0],sig_projs[:,1],sig_projs[:,2],s=5, c='grey', alpha=0.05)
        
    ax.scatter(points_neut[:,0],points_neut[:,1],points_neut[:,2],s=10,c='blue')
    ax.scatter(points_food[:,0],points_food[:,1],points_food[:,2],s=10,c='green')
    ax.scatter(points_shock[:,0],points_shock[:,1],points_shock[:,2],s=10,c='red')




# function to plot pure clusters projected along significant wavelet dimensions
# significant dims are chosen from impact value on SHAP test, clusters from purity measure vs context
# MULTISESSION VERSION
# INPUTS: data - list of full wavelet projections, sig_features - list of 3 features to use as plot axis
# sig_clusters - list of lists of three clusters from label set to plot (one for each context)
# labels - list of vectors of cluster labels 
def plot_sig_projs_all(data, sig_features, sig_clusters, labels, plot_all=False, norm=False):
    
    #sig_projs = data[:,sig_features]
    
    if norm:
        #TODO: standardize data
        print('Not yet implemented')
    
    points_neut = []
    points_food = []
    points_shock = []
    
    for i in range(len(data)):
        sig_projs = data[i][:,sig_features]
        points_neuti = sig_projs[labels[i] == sig_clusters[i][0]]
        points_foodi = sig_projs[labels[i] == sig_clusters[i][1]]
        points_shocki = sig_projs[labels[i] == sig_clusters[i][2]]
        
        points_neut.append(points_neuti)
        points_food.append(points_foodi)
        points_shock.append(points_shocki)
    
    
    points_neut = np.vstack(points_neut)
    points_food = np.vstack(points_food)
    points_shock = np.vstack(points_shock)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # TODO: fix axis labels and ranges
    
    if plot_all:
        all_data = np.vstack(data)
        sig_projs = all_data[:,sig_features]
        ax.scatter(sig_projs[:,0],sig_projs[:,1],sig_projs[:,2],s=5, c='grey', alpha=0.05)
        
    ax.scatter(points_neut[:,0],points_neut[:,1],points_neut[:,2],s=10,c='blue',label='Neutral')
    ax.scatter(points_food[:,0],points_food[:,1],points_food[:,2],s=10,c='green',label='Food')
    ax.scatter(points_shock[:,0],points_shock[:,1],points_shock[:,2],s=10,c='red',label='Shock')
    
    ax.legend()
    #remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # add axis labels
    ax.set_xlabel('Wavelet A')
    ax.set_ylabel('Wavelet B')
    ax.set_zlabel('Wavelet C')
    
    # If you want just the points in abstract space
    #ax.set_axis_off()





# function for computing a context "purity" measure
# in this case, it measures how pure a cluster is for any of the three contexts
# contexts are numerically labelled: (1: neutral, 2: food, 3: shock, 0: below threshold/error)
# INPUTS: labels - cluster labels (samples x 1, cluster labels), contexts - array of context frame ranges
# mode - '3_contexts' for default 3 context setup, 'single' otherwise
# NOTE: will return empty of neither mode is selected (should probable throw an error, not implemented yet)
def compute_context_purity(labels, contexts, ignore_noise=False, mode='3_contexts'):
    
        
    if ignore_noise:
        first_clust = 0
    else:
        first_clust = -1
    
    out_stats = []
    
    
    
    if mode == '3_contexts':
        for i in range(first_clust, max(labels)+1):
        
             
            clust_b = labels == i
            if np.count_nonzero(clust_b):
                clust_size = np.count_nonzero(clust_b)
        
                neut_b = contexts == 0
                comp_n = clust_b & neut_b
        
                n_overlap = np.count_nonzero(comp_n)
                n_purity = n_overlap / clust_size
        
                food_b = contexts == 1
                comp_f = clust_b & food_b
        
                f_overlap = np.count_nonzero(comp_f)
                f_purity = f_overlap / clust_size
        
                shock_b = contexts == 2
                comp_s = clust_b & shock_b
        
                s_overlap = np.count_nonzero(comp_s)
                s_purity = s_overlap / clust_size
            else:
                n_purity = 0
                f_purity = 0
                s_purity = 0
        
            clust_stats = [i, n_purity, f_purity, s_purity]
        
            out_stats.append(np.array(clust_stats))
            
    elif mode == 'single':
        for i in range(first_clust, max(labels)+1):
            
            clust_b = labels == i
            clust_size = np.count_nonzero(clust_b)
            
            context_b = contexts == 1
            comp_c = clust_b & context_b
            
            c_overlap = np.count_nonzero(comp_c)
            c_purity = c_overlap / clust_size
            r_purity = c_overlap / np.count_nonzero(context_b)
            
            clust_stats = [i, c_purity, r_purity]
            
            out_stats.append(np.array(clust_stats))
    
    
    return out_stats


# function to compute the purities for two sets of labels
# INPUTS: true_labels - 1st set of cluster labels(samples x 1), input_labels - 2nd set of cluster labels(samples x 1)
# OUPUTS: out_purities - list of lists of purities
def compute_labelset_purity(true_labels,input_labels,ignore_noise=False):
    
    true_labels = true_labels.astype(int)
    input_labels = input_labels.astype(int)
    
    if ignore_noise:
        first_clust = 0
    else:
        first_clust = -1
    
    out_purities = []
    
    for i in range(first_clust, max(input_labels)+1):
        out_stats = []
        for j in range(first_clust, max(true_labels)+1):
            
            clust_b = true_labels == j
            clust_size = np.count_nonzero(clust_b)
            
            input_clust = input_labels == i
            comp_c = clust_b & input_clust
            
            c_overlap = np.count_nonzero(comp_c)
            c_purity = c_overlap / clust_size
            if np.count_nonzero(input_clust):
                r_purity = c_overlap / np.count_nonzero(input_clust)
            else:
                r_purity=0
            
            clust_stats = [j, c_purity, r_purity]
            out_stats.append(np.array(clust_stats))
        out_purities.append(out_stats)
    
    return out_purities
    

# computes overall context "purity" for a set of cluster labels
# INPUTS: purities - list of context purities generated from compute_context_purity
def compute_overall_purity(purities):
    
    total_p = 0
    num_clusts = len(purities)
    
    for clust in purities:
        
        max_p = np.max(clust[1:])
        
        total_p = total_p + max_p
        
    overall_purity = total_p / num_clusts
    
    return overall_purity


# computes maximum purity for a set of cluster labels for optimizing a particular cluster match
# INPUTS: purities = list of context(label) purities generated from compute_context_purity
# comp - which direction you want to maximise the purity over(clust=proportion of cluster, context = proportion of context)
def compute_max_purity(purities, comp = 'clust'):
    
    max_p = 0
    max_pair = 0
    if comp=='clust':
        for clust in purities:
            max_c = clust[1]
            if max_c > max_p:
                max_p = max_c
                max_pair = clust[1:]
    elif comp=='context':
        for clust in purities:
            max_c = clust[2]
            if max_c > max_p:
                max_p = max_c
                max_pair = clust[1:]
    return max_pair

# function for extracting max purities for each label from a labelset purity list
# INPUTS: purities = list of lists of purities for labelset comparison(len(2nd labels) x len(1st labels))
# OUPUTS: max_purities - list of max purities
def compute_max_labelset_purities(purities, comp='clust'):
    
    max_purities = []
    for i in range(0,len(purities)):
        curr_max_pair = compute_max_purity(purities[i],comp)
        
        max_stats = [i,curr_max_pair[0],curr_max_pair[1]]
        max_purities.append(np.array(max_stats))
        
    return max_purities


# function for extracting the top purities for each context
# INPUTS: purities = list of arrays of context purities (from compute_context_purity)
# OUTPUTS: max_purities = list of maximum purity values for each context
def compute_max_context_purities(purities):    
    
    max_n = 0
    max_f = 0
    max_s = 0
    for i in range(0,len(purities)):
        
        curr_max_n = purities[i][1]
        curr_max_f = purities[i][2]
        curr_max_s = purities[i][3]
        
        if curr_max_n > max_n:
            max_n = curr_max_n
        if curr_max_f > max_f:
            max_f = curr_max_f
        if curr_max_s > max_s:
            max_s = curr_max_s
    
    max_purities = [max_n, max_f, max_s]
    return max_purities



# function for plotting purities as a bar graph
# INPUTS: purities - list or array of context purities, cluster - int as index of the cluster to plot
def plot_context_purities_single(purities, cluster):
    
    fig, ax = plt.subplots()
    
    #plot params
    num_contexts = np.size(purities,1)
    bar_locs = np.arange(num_contexts)
    
    ax.bar(bar_locs,purities[cluster,:],width=0.4,color=['b','g','r'])
    ax.set_xticks([])


def plot_context_purities(purities):
    
    fig, ax = plt.subplots()
    purities = np.array(purities)
    # plot params
    num_clusts = np.size(purities,0)
    width = 0.25
    bar_locs = np.arange(num_clusts)
    neut_rects = ax.bar(bar_locs-width,purities[:,1],width=width,label='neut',color='b')
    food_rects = ax.bar(bar_locs,purities[:,2],width=width,label='food',color='g')
    shock_rects = ax.bar(bar_locs+width,purities[:,3],width=width,label='shock',color='r')
    
    ax.set_ylabel('Purities')
    ax.set_xticks([])
    #ax.grid(axis='y',zorder=-1)
    ax.legend()


# function for plotting mean maximum context purities as a bar plot (1 bar for each context)
# INPUTS: purities = list of max context purities for each session    
def plot_mean_context_purities(purities):
    
    fig, ax = plt.subplots()
    mean_purities = np.mean(purities,0)
    #std_purities = np.std(purities,0)
    sem_purities = sem(purities,0)
    
    num_contexts = np.size(purities,1)
    bar_locs = np.arange(num_contexts)
    ax.bar(bar_locs,mean_purities,yerr=sem_purities,error_kw=dict(lw=5, capsize=5, capthick=3),width=1,color=['b','g','r'])
    ax.set_xticks([])
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    ax.yaxis.set_tick_params(labelsize=20)
    

# function for plotting responsiveness ratios as a bar plot (one bar for each of teh two conditions)
# INPUTS: responsiveness ratios for each condition (currently expects two conditions)
def plot_responsive_ratio_bar(r_ratios):
    
    fig, ax = plt.subplots()
    mean_ratios = np.mean(r_ratios,1)
    sem_ratios = sem(r_ratios,1)
    
    num_ratios = np.size(r_ratios,0)
    bar_locs = np.arange(num_ratios)
    ax.bar(bar_locs,mean_ratios,yerr=sem_ratios,error_kw=dict(lw=5, capsize=5, capthick=3),width=0.8,color=['lightblue','purple'])
    ax.set_xticks([])
    ax.set_ylim([0,0.5])
    ax.set_yticks([0,0.25,0.5])
    ax.yaxis.set_tick_params(labelsize=20)


# helper to turn ranges of frames to a label vector
# INPUTS: ranges - array (context by 2) of frame ranges for contexts (order is assumed fixed: neutral, food, shock)
def range_to_label(ranges):
    
    samples = np.max(ranges)
    
    labels = np.zeros(samples)
    
    labels[ranges[0,0]:ranges[0,1]] = 1
    labels[ranges[1,0]:ranges[1,1]] = 2
    labels[ranges[2,0]:ranges[2,1]] = 3
    
    return labels.astype(int)



    
# function for computing point distance to cluster centres
# This could be modified to take cluster centre as centre of exemplars, and then calc all member point distances
def calc_centroid_dists(data):
    
    centroid = np.mean(data, axis=0)
    samples = np.size(data,0)
    dists = np.zeros(samples)
    for i in range(samples):
        dists[i] = np.sqrt((data[i,0]-centroid[0])**2 + (data[i,1]-centroid[1])**2)
    
    return dists
    
    
# function for turning a one-hot set of labels to a single dim label vector    
def one_hot_to_vector(labels):
     
    out = np.zeros(np.size(labels,axis=0))
     
    for i in range(np.size(labels,axis=1)):
         
        out[labels[:,i]==1] = i
    
    return out



# function for plotting a 3d feature over a set number of points(time)
# INPUTS: data - array (samples x 3) of feature locations
def plot_feature_3d(data):
    
    n_samples = np.size(data,0)
    fig = plt.figure()
    axs = fig.subplots(3,1,sharey= True,sharex=True)
    #ax = axs.ravel()
    
    samples = range(n_samples)
    
    axs[0].plot(samples,data[:,0])
    axs[0].yaxis.set_ticklabels([])
    axs[0].set_ylabel('X',fontsize=20)
    axs[1].plot(samples,data[:,1])
    axs[1].set_ylabel('Y',fontsize=20)
    axs[2].plot(samples,data[:,2])
    axs[2].set_ylabel('Z',fontsize=20)
    
    axs[2].set_xlabel('Bins',fontsize=20)
    axs[2].xaxis.set_tick_params(labelsize=20)
    
    for ax in axs:
        ax.label_outer()
        
    fig.subplots_adjust(hspace=0)






# Function to plot the r2 values from s single glms run as sorted bars
def plot_r2s(r2s, threshold=False, colour='blue', prod=False):
    
    s_vals = r2s
    s_vals = np.array(s_vals)
    # Trim any values of 1 (these are a result of an underflow in the glm run)
    oor_mask = s_vals < 0.99
    s_vals = s_vals[oor_mask]
    
    
    if threshold:
        mask = s_vals > threshold
        s_vals = s_vals[mask]
        
    s_vals = sorted(s_vals,reverse=True)
    
    bar_locs = range(len(s_vals))
    
    fig, ax = plt.subplots()
    
    ax.bar(bar_locs, s_vals, color=colour)
    
    if prod:
        
        ax.set_ylim([0,0.7])
        #ax.set_xlim([-5,180])
        
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    
    return fig




# Function to plot the r2 values from 3 single glm runs as unsorted lines
# designed to take 3 matched length sets of r2 values (same targets) resulting
# from 3 different predictors
def plot_r2s_lines(r2s_one, r2s_two, r2s_three, threshold=False, prod=False):
    
    s_val1 = r2s_one
    oor_mask = s_val1 < 0.99
    s_val1 = s_val1[oor_mask]
    
    s_val2 = r2s_two
    oor_mask = s_val2 < 0.99
    s_val2 = s_val2[oor_mask]
    
    s_val3 = r2s_three
    oor_mask = s_val3 < 0.99
    s_val3 = s_val3[oor_mask]
    
    if threshold:
        mask = s_val1 > threshold
        s_val1 = s_val1[mask]
        
        mask = s_val2 > threshold
        s_val2 = s_val2[mask]
        
        mask = s_val2 > threshold
        s_val2 = s_val2[mask]
        
    
    fig, ax = plt.subplots()
    
    ax.plot(s_val1, linewidth=2, color='blue')
    
    ax.plot(s_val2, linewidth=2, color='orange')
    
    ax.plot(s_val3, linewidth=2, color='red')
    
    if prod:
        
        ax.set_ylim([0,0.7])
        #ax.set_xlim([-5,180])
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        
    return fig




# Helper function for extracting specific time periods
# data and label set needs to be the same length
def extract_bins(data, label):
    
    mask = label > 0
    outdata = data[mask].copy()
    
    return outdata
    
# Helper pre-processing function to cut out noise periods from labelsets and
# accompanying numerical data
# labels and num_data need to be the same length, asumes labels are a vector
# assumes noise is labelled as -1 in labels
def precut_noise(labels, num_data):
    
    # get mask of non-noise labels
    mask = labels >= 0
    outlabels = labels[mask].copy()
    
    # check for dim of numerical data
    if num_data.ndim == 1:
        outdata = num_data[mask].copy()
    else:
        outdata = num_data[mask,:].copy()
    
    return outlabels, outdata
    
    
