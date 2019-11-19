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
from lib import network
from lib import vision


class Writer():
    def __init__(self, data_name, txt_name, append = False):
        log_dir = './log/'+data_name
        util.create_dir(log_dir)
        self.log_file = log_dir + '/'+txt_name+".txt"
        
        ## write header of the file
        if append:
            with open(self.log_file, 'a') as f:    
                f.write(str(datetime.datetime.now())+'\n')
        else:
            with open(self.log_file, 'w') as f:    
                f.write(str(datetime.datetime.now())+'\n')
                
    def write(self, string):
        with open(self.log_file, 'a') as f:
            f.write(string)    


class Batch_stream():
    def __init__(self):
        pass
        
    def iterator(self):
        pass
 
class Network():
    def __init__(self):
        self.network_name = 'Network'
    
    def _build(self):
        raise NotImplementedError

    def _train(self):
        raise NotImplementedError
