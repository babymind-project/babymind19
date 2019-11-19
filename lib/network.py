from __future__ import division

import datetime
import glob
import traceback

import IPython
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tflearn
from tflearn.layers.core import fully_connected, flatten
from tflearn.layers.conv import conv_2d, conv_2d_transpose
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization

from lib import util


"""
one layer
""" 
def conv(h_0,filters,kernel_size,strides, activation = 'relu', trainable = True, batch_norm = True):
    init = tflearn.initializations.truncated_normal (shape=None, mean=0.0, stddev=0.01, dtype=tf.float32, seed=None)
    h1 = conv_2d(h_0, nb_filter = filters, filter_size = kernel_size, strides = strides, weights_init = init, trainable = trainable) 
    if batch_norm:
        h1_bn = batch_normalization(h1, trainable = trainable)
    else:
        h1_bn = h1
    # conv layer should have batch norm
    if activation == 'relu':
        h1_o = tf.nn.relu(h1_bn)
    elif activation == 'none': 
        h1_o = h1_bn
    print(h1_o.shape)
    return h1_o

def deconv(h_0,filters, kernel_size, strides, output_shape = [], activation = 'relu', trainable = True):
    if not output_shape:
        h = int(  ((h_0.shape[1] - 1) * strides) + kernel_size  ) - 1
        w = int(  ((h_0.shape[2] - 1) * strides) + kernel_size  ) - 1
    else:
        h = int(output_shape[1])
        w = int(output_shape[2])    
    init = tflearn.initializations.truncated_normal (shape=None, mean=0.0, stddev=0.01, dtype=tf.float32, seed=None)
    h1 = conv_2d_transpose(h_0, nb_filter = filters, filter_size = kernel_size, strides = strides, output_shape = [h,w], weights_init = init, trainable = trainable) 
    h1_bn = batch_normalization(h1, trainable = trainable)
    if activation == 'relu':
        h1_o = tf.nn.relu(h1_bn)
    elif activation == 'none': 
        h1_o = h1_bn
    print(h1_o.shape)
    return h1_o

"""
complex layer
"""
def conv_deconv(img, trainable = True, net = {}):
    ###to deconv
    net['h10'] = conv(img,filters=32,kernel_size=3,strides=1, trainable = trainable)
    net['h11'] = conv(net['h10'],filters=64,kernel_size=3,strides=2, trainable = trainable)
    ###to deconv
    net['h20'] = conv(net['h11'],filters=64,kernel_size=3,strides=1, trainable = trainable)
    net['h21'] = conv(net['h20'],filters=128,kernel_size=3,strides=2, trainable = trainable)
    ###to deconv
    net['h30'] = conv(net['h21'],filters=128,kernel_size=3,strides=1, trainable = trainable)
    net['h31'] = conv(net['h30'],filters=256,kernel_size=3,strides=2, trainable = trainable)
    ###to deconv
    net['h40'] = conv(net['h31'],filters=256,kernel_size=3,strides=1, trainable = trainable)
    net['h41'] = conv(net['h40'],filters=512,kernel_size=3,strides=2, trainable = trainable)
    ###to deconv
    net['h50'] = conv(net['h41'], filters=512,kernel_size=3,strides=1, trainable = trainable)                 
    net['h51'] = conv(net['h50'],filters=1024,kernel_size=3,strides=2, trainable = trainable)
    ###to deconv

    net['embedding'] = conv(net['h51'],filters=1024,kernel_size=3,strides=1, trainable = trainable, batch_norm = True)
    
    net['d5']   = deconv(net['embedding'],filters=512,kernel_size=3,strides=2, output_shape = net['h50'].shape, trainable = True)
    net['d4_i'] = merge([net['d5'], net['h50']], mode= 'concat', axis = -1)
    net['d4']   = deconv(net['d4_i'], filters=256, kernel_size=3, strides=2, output_shape = net['h40'].shape, trainable = True)
    net['d3_i'] = merge([net['d4'], net['h40']], mode= 'concat', axis = -1 )
    net['d3']   = deconv(net['d3_i'], filters=128, kernel_size=3, strides=2, output_shape = net['h30'].shape, trainable = True)
    net['d2_i'] = merge([net['d3'], net['h30']], mode= 'concat', axis = -1)
    net['d2']   = deconv(net['d2_i'], filters=64, kernel_size=3, strides=2, output_shape = net['h20'].shape, trainable = True)
    net['d1_i'] = merge([net['d2'], net['h20']], mode= 'concat', axis = -1)
    net['out']  = deconv(net['d1_i'], filters=32, kernel_size=3, strides=2, trainable = True)  
    return net

"""
network
"""

def u_net(frame, output_ch, scope_name = 'u_net', trainable = True, reuse = False):
    with tf.variable_scope(scope_name, reuse = reuse):
        init = tflearn.initializations.truncated_normal(shape=None, mean=0.0, stddev=0.01, dtype=tf.float32, seed=None)
        conv_deconv_output = conv_deconv(frame, trainable = trainable)
        out = conv_deconv_output['out']
        embed = conv_deconv_output['embedding']
        mask_logit = conv_2d(out, nb_filter = output_ch, filter_size = 1, strides = 1, padding = 'same', weights_init = init, trainable = trainable) 
        #obj_mask = tf.nn.sigmoid(obj_mask) # -> why using 'tanh', 'sigmoid' do not make all values go zero?
        mask_sigmoid = tf.nn.sigmoid(mask_logit)
        return mask_logit, mask_sigmoid, embed

def pose_net(embed, mask_ch, scope_name ='pose_net', reuse = False):
    with tf.variable_scope(scope_name, reuse = reuse):
        # fully connected layer should not have batchnorm
        # do not use relu!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # relu cannot do overfitting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # batch norm requires at least 2 batch size
        embed = fully_connected(embed, 4096, activation = None)
        embed = batch_normalization(embed)
        embed = tf.nn.relu(embed)
        
        embed = fully_connected(embed, 1024, activation = None)
        embed = batch_normalization(embed)
        embed = tf.nn.relu(embed)

        se3_T = fully_connected(embed, 512, activation = tf.nn.sigmoid) # relu of the last layer makes diverge?
        se3_T = fully_connected(se3_T, mask_ch*3, activation = None)
        
        se3_R = fully_connected(embed, 512, activation = tf.nn.sigmoid)
        se3_R = fully_connected(se3_R, mask_ch*3, activation = None)
        
        se3 = tf.concat([se3_T,se3_R], axis = 1)
        se3 = tf.reshape(se3, [-1, mask_ch, 6])   
        return se3
    








