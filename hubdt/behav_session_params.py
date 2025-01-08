#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:03:55 2021

@author: proxy_loken

Default Session Parameter Loading and Storage for Behavioural/Neural Classification 

Important: Session parameters should only be changed by the "load_session_params"
function. This should be called whenever the current session is changed. Data loading
and other modules that need these params can import this file dynamically
"""

# SESSION PARAMETERS
# Default values (first session) demonstrate how to setup the config file for loading parameters
# these default values can be used directly as an example


session_name = 'I_Blocks_231'

animal_id = 'I'

session_date = '0330'

#signal frequency (bin rate in this case)
sfreq = 5

# video framerate
vid_fr = 15

# File Params
tracking_mat = 'i231_3ddata.mat'

stmtx_mat = '03_30_I_Blocks231_cat_data.mat'

behav_mat = 'encoding_0330Ib231_5fps.mat'

block_resid_mat = 'i231_block_average_resids.mat'


# STMTX Params
# Time in seconds, binned at 200ms (in this example)
recording_start = 0
start_bin = 2242


# Video Params
# Frame counts at 15fps (Note: This offset will be needed for anything from 
# raw frames, ie limb position and frame indices. Behavioural annotations should
# be pre-trimmed before loading)
frame_start = 379

# Block Params
# binned at 200ms, these are counting from the recording start (bins before "start_bin" are pre-trimmed)
neutral_start = 18229
neutral_end = 24811

food_start = 1627
food_end = 8206

shock_start = 9974
shock_end = 16505





# Proper session parameter loading

import configparser

config_file = 'behav_session_config.cfg'


class SessionParams:
    def __init__(self, params_dict):
        for k, v in params_dict.items():
            setattr(self, k, v)


# helper function for float checking
def will_it_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False






# SESSION LOADING FUNCTIONS
# Load session parameters from file (config file)
# INPUTS: sess_name - short form name of the session as a string
# OUTPUTS: curr_sess - a session parameter object containing all the session parameters
def load_session_params(sess_name):
    
    cfg = configparser.RawConfigParser()
    cfg.read(config_file)
    # make a dictionary of the loaded parameters
    params = dict(cfg.items(sess_name))
    # convert digits to ints, other numbers to floats
    for k, v in params.items():
        if v.isdigit():
            params[k] = int(v)
        elif will_it_float(v):
            params[k] = float(v)
            
        if v=='False':
            params[k] = False
    
    # instantiate SessionParams class with parameters
    curr_sess = SessionParams(params)    
    
    return curr_sess



# helper to pull session names from the config file
def load_session_names():
    cfg = configparser.RawConfigParser()
    cfg.read(config_file)
    # dict of session names
    names = dict(cfg.items())
    
    return names
