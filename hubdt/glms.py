#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:30:46 2022

@author: proxy_loken
"""
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import r2_score

from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor


# function to log transform and standardize input data
def log_scale_transform(data):
    
    log_scale_transformer = Pipeline([('log_transform', FunctionTransformer(np.log, validate=False)), ('scaler', StandardScaler())])
    out_data = log_scale_transformer.fit_transform(data)
    out_data = np.nan_to_num(out_data,nan=0)
    return out_data


# function to standardize the data
def standardize(data):
    
    std = StandardScaler()
    out_data = std.fit_transform(data)
    
    return out_data


# Function to preprocess data for use in GLM
# Inputs can be categories or numerical
def preprocess_data(pred_data, input_cats=False, keep_noise=True):
    
        
    if input_cats:
        oh = OneHotEncoder(sparse=False)
        oh_data = oh.fit_transform(pred_data.reshape(-1,1))
        # Drop the first category (noise/-1 values) if desired
        if not keep_noise:
            oh_data = oh_data[:,1:]
    else:
        oh_data = standardize(pred_data)
    
    return oh_data


# Function to binarize label data for use in a GLM. 
def preprocess_targets(targets, target_cats=True, keep_noise=True):
    
    # Binarize the target labels
    # Note: for Context labels, this will result in 4 classes: None, Neutral, Food, Shock
    if target_cats:
        lb = LabelBinarizer()
        tars = lb.fit_transform(targets)
    
    if not keep_noise:
        tars = tars[:,1:]
        
    return tars
    



# Wrapper for splitting dataset into train and test
def split_data(X,y, size=.1):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    
    return X_train, X_test, y_train, y_test



# Function to generate a set of scores for a given estimator
def score_estimator(estimator, X_test, y_test):
    
    y_pred = estimator.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    # Poisson deviance can't be calculated for negative values, so we mask those out
    mask  = y_pred > 0
    if (~mask).any():
        print("Warning: Estimator gave non-positive predictions. These are ignored when computing Poisson Deviance")
        
    mpd = mean_poisson_deviance(y_test[mask],y_pred[mask])
    
    return mse,mae,mpd

# Function to calculate r2s for a given estimator
# Type argument determines output shape(vector or scalar)
def calc_r2s(estimator, X_test, y_test, out_type='raw'):
    
    y_preds = estimator.predict(X_test)
    
    if y_test.ndim > 1:
        if out_type=='raw':
            r2s = r2_score(y_test, y_preds, multioutput='raw_values')
        elif out_type=='mean':
            r2s = r2_score(y_test, y_preds, multioutput='uniform_average')
        elif out_type=='max':
            r2s = r2_score(y_test, y_preds, multioutput='raw_values')
            r2s = max(r2s)
        elif out_type=='sum':
            r2s = r2_score(y_test, y_preds, multioutput='raw_values')
            r2s = sum(r2s)
    else:
        r2s = r2_score(y_test, y_preds)
        
    return r2s
        
        
    
    


# Wrapper for the estimator instantiation and fit
# Generates a multioutput regressor if the target y has more than two categories
def fit_glm(X_train, y_train, rtype='Poisson', alpha=0, fit_intercept=True, m_iter=300):
    
    if rtype == 'Poisson':
        estimator = PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter = m_iter)
    else:
        print('Estimator not Supported')
        return
    
    if y_train.ndim > 1:
        m_regr = MultiOutputRegressor(estimator)
        m_regr.fit(X_train, y_train)
        
        return m_regr
    else:    
        estimator.fit(X_train, y_train)    
        return estimator


# Helper to generate residuals from a fit GLM
def generate_residuals(estimator, X, y,single=False):
    
    if single:
        X = X.reshape(-1,1)
    trans_data = estimator.predict(X)
    resids = y-trans_data
    
    return resids
    


def fit_single_glms(X_train,y_train, full_data, full_tars, rtype='Poisson', m_iter=300, single_tars=False):
    
    if rtype == 'Poisson':
        estimator = PoissonRegressor(max_iter = m_iter)
    else:
        print('Estimator not Supported')
        return
    
    
    # TODO: rewrite to allow for raw, summed,or max scores from multioutut
    if single_tars:
        
        curr_r2s = []
        curr_resids = []
        
            
    
    
    
    
    if y_train.ndim > 1:
        regr = MultiOutputRegressor(estimator)
    else:
        regr = estimator
    
    r2s = []
    resids = []    
    
    for i in range(np.shape(X_train)[1]):
        
        X = X_train[:,i]
        X = X.reshape(-1,1)
        regr.fit(X,y_train)
        
        r2s.append(regr.score(X,y_train))
        resids.append(generate_residuals(regr, full_data[:,i], full_tars,single=True))
        
    
    return r2s, resids
        
        

    
# UTILITIES

# Function to calculate the proportion of significant contributors to a single glms run
def calc_proportions(r2s,threshold=0.01):
    
    total = len(r2s)
    r2s = np.array(r2s)
    mask = r2s > threshold
    num_sig = np.sum(mask)
    
    prop = num_sig/total
    
    
    return num_sig, prop







    




    