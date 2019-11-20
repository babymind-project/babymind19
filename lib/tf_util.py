import datetime
import os
import shutil
import yaml

import IPython
import numpy as np
import tensorflow as tf
from pathlib import Path

def collect_variables(scope_list):
    '''
    scope_list : list of str(scope) (ex: ['a', 'b', 'c'])
    '''
    vars = []
    for scope in scope_list:
        vars+= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return vars

def print_tensor(x):
    print_op = tf.print(x)
    with tf.control_dependencies([print_op]):
        x = x+0.
    return x
