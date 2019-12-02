import glob
import os

import IPython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import io
from scipy.signal import medfilt
from scipy.optimize import minimize

from lib.math import *
from lib import util
from lib import vision

def load_vicon(vicon_dir):
    vicon_files = sorted(glob.glob('*.npy'))
    
    traj = []
    for f in vicon_files:
        se3 = np.load(f)
        traj.append(np.expand_dims(se3,0))
    return np.concatenate(traj,0)


def compare(config):
    '''
    vicon data is required!
    execute 'read_bag' first
    '''

    task_name = config['task_name']
    fps = config['animation']['fps']
    object_list = util.load_txt('./configure/%s_objects.txt'%task_name)
    nb_object = len(object_list)

    data_dir = './data/'+task_name
    pose_dir = './output/'+task_name+'/read_pose'
    output_dir = './output/'+task_name+'/compare_traj'
    util.create_dir(output_dir, clear = True)
    
    ## load data
    demo_list = os.listdir('./data/'+task_name)
    for demo in demo_list:
        rgb_demo_dir = data_dir+'/'+demo+'/rgb'
        cam_demo_dir = data_dir+'/'+demo+'/vicon/k_zed'        
        assert len(object_list) == 1 ## to do : multiple object
        obj_demo_dir = data_dir+'/'+demo+'/vicon/'+object_list[0] 
        se3_vision = np.load(pose_dir+'/'+demo+'/pose_traj.npy')
        se3_vicon = load_vicon(obj_demo_dir)
        se3_cam = laod_vicon(cam_demo_dir)

        IPython.embed()
        sys.exit()

        obj_vicon = vision.SE3object(np.zeros(6), angle_type = 'axis')
        obj_vision = vision.SE3object(np.zeros(6), angle_type = 'axis')

        
        