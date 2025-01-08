#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 08:04:08 2022

@author: proxy_loken

adapted from Qi et al, (2017) and examples at keras.io

Point net model setup and evaluation

"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers








# MODEL BUILDING FUNCTIONS

# Convolution layer
def conv_bn(x, filters):
    
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    
    return layers.Activation("relu")(x)

# Dense layer
def dense_bn(x, filters):
    
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    
    return layers.Activation("relu")(x)

# Regulizer (this constrains the transformation to be near to orthogonal)
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


# Transformer net constructor
def tnet(inputs, num_features, units=32):
    
    # initialize bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_bn(inputs, units)
    x = conv_bn(x, units*2)
    x = conv_bn(x, units*16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units*8)
    x = dense_bn(x, units*4)
    x = layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=bias, activity_regularizer=reg)(x)
    
    feat_T = layers.Reshape((num_features, num_features))(x)
    # apply affine transform to the input features
    return layers.Dot(axes=(2,1))([inputs, feat_T])


# Point net constructor
def point_net(num_points, num_classes, units=32, dims=1):
    
    inputs = keras.Input(shape=(num_points, dims))
    
    x = tnet(inputs, dims, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = tnet(x, units, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units*2)
    x = conv_bn(x, units*16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units*8)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, units*4)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    model.summary()
    
    return model


# ABLATION versions of Point Net
def point_net_no_transform(num_points, num_classes, units=32, dims=1):
    
    inputs = keras.Input(shape=(num_points, dims))
    
    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units*2)
    x = conv_bn(x, units*16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units*8)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, units*4)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnotransform')
    model.summary()
    
    return model


def point_net_no_pool(num_points, num_classes, units=32, dims=1):
    
    inputs = keras.Input(shape=(num_points, dims))
    
    x = tnet(inputs, dims, units)
    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = tnet(x, units, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units*2)
    x = conv_bn(x, units*16)
    x = layers.Flatten()(x)
    x = dense_bn(x, units*8)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, units*4)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnopool')
    model.summary()
    
    return model


def point_net_no_pool_no_transform(num_points, num_classes, units=32, dims=1):
    
    inputs = keras.Input(shape=(num_points, dims))

    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units*2)
    x = conv_bn(x, units*16)
    
    # take global signal either sequentially: RNN, or all at once: flatten
    x = layers.Flatten()(x)
    #cell = layers.SimpleRNNCell(units)
    #x = layers.RNN(cell)(x)
    
    x = dense_bn(x, units*8)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, units*4)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnopoolnotransform')
    model.summary()
    
    return model



# point net segmentation constructor
def point_net_segment(num_points, num_classes, units=32, dims=1):

    inputs = keras.Input(shape=(num_points, dims))
    
    transform_1 = tnet(inputs, dims, units)
    features_1 = conv_bn(transform_1, units)
    features_2 = conv_bn(features_1, units*2)
    features_3 = conv_bn(features_2, units*2)
    transform_2 = tnet(features_3, units*2, units)
    features_4 = conv_bn(transform_2, units*8)
    features_5 = conv_bn(features_4, units*16)
    global_1 = layers.MaxPool1D(pool_size=num_points)(features_5)
    
    #global_2 = keras.ops.tile(global_1, [1, num_points, 1])
    #global_2 = keras.layers.Lambda(keras.backend.tile, arguments={'n':[1,num_points,1]})(global_1)
    global_2 = tf.tile(global_1, [1, num_points, 1])
    
    segmentation_input = layers.Concatenate(name='segmentation_input')(
        [
         features_1,
         features_2,
         features_3,
         transform_2,
         features_4,
         global_2            
            ]
        )
    
    segment_1 = conv_bn(segmentation_input, units*2)
    
    outputs = layers.Conv1D(num_classes, kernel_size=1, activation='softmax', name='segmentation_head')(segment_1)
        
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetsegment')
    model.summary()
    
    return model


    

# Visualisation Functions

# run critical points
def predict_critical(model, samples, layer_name='activation_31'):
    
    # grab the last conv layer before the maxpooling
    #layer_name = 'conv1d_21'
    #layer_name = 'activation_31'
    # setup a new model that outputs the predictions at the target layer
    layer = model.get_layer(name=layer_name)
    crit_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    # feed in the sample inputs
    crit_preds = crit_extractor.predict(samples)    
    
    return crit_preds


# run upper points
def predict_upper(model, samples, layer_name = 'global_max_pooling1d_5'):
    
    # grab the output of the last max pooling layer
    #layer_name = 'global_max_pooling1d_5'
    # setup a new model that outputs the predictions at the target layer
    layer = model.get_layer(name=layer_name)
    upper_extractor = keras.Model(inputs=model.inputs, output=layer.output)
    # feed in the sample inputs
    upper_preds = upper_extractor.predict(samples)    
    
    return upper_preds


# generate critical points set, also returns mean dimensionality of the critical set
# INPUTS: crit_preds: predictions of last conv layer on input samples, num_samples: number of samples 
# to generate critical points for, samples: samples inputs used to generate predictions
def generate_critical(crit_preds, num_samples, samples):
    
    # index points that contribute to maxpooling
    cs_index = np.argmax(crit_preds, axis=1)
    # list to hold critical points
    cs=[]
    dims=[]
    # extract corresponding critical points from samples used to generate predictions 
    for i in range(num_samples):
        temp=[]
        dims.append(np.size(np.unique(cs_index[i])))
        for index in cs_index[i]:
            temp.append(samples[i][index])
        cs.append(temp)
    # convert to numpy array
    cs = np.array(cs)
    # take mean of dims
    mean_dim = np.mean(dims)    
    
    return cs, mean_dim


# generate upper points set
# INPUTS: max_preds: maxpool layer predictions on samples, max_unit_preds: conv layer predictions
# on unit sphere input, num_samples: number of samples to generate for, samples: samples used to
# generate predictions, unit_sphere: unit sphere points used to generate max output predictions
def generate_upper(max_preds, max_unit_preds, num_samples, samples, unit_sphere):
    # TODO: Check shapes, reshape if needed
    ups = []
    # extract points that do not change the max of the last conv layer
    for i in range(num_samples):
        temp = []
        # get difference between conv output on unit sphere(max_unit_preds) and maxpool output on samples (max_preds)
        x = max_preds[i] - max_unit_preds
        x = np.min(x, axis=1)
        for j in range(x.shape[0]):
            if (x[j] >= 0):
                temp.append(unit_sphere[j])
        ups.append(temp)
        
    ups = np.array(ups)
    
    return ups














