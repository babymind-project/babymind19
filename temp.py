import argparse
import glob
import os
import signal
import subprocess
import sys
from termcolor import colored
import time
import traceback

import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rosbag

from lib import util
from lib import vision
from lib.math import *

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('task_name')
        parser.add_argument('--clear', action = 'store_true')
    
        args = parser.parse_args()
        task_name = args.task_name
        CLEAR = args.clear
    except:
        traceback.print_exc()
        print(colored('type like: python3 ./read_bag.py [task_name] [option]'),'r')
        print(colored('ex: python3 ./read_bag.py task1 --clear'))
    
    task_dir = './data/'+task_name
    demo_dirs = sorted(os.listdir(task_dir))
    
    config = util.load_yaml('./configure/'+task_name+'.yaml')
    fps = config['animation']['fps']
        
    object_names = util.load_txt('./configure/'+task_name+'_objects.txt')
    object_names += ['k_giant_ur3', 'k_zed']
    for demo_name in demo_dirs:
        se3_dict = {}
        for obj in object_names:
            vicon_dir = './data/'+task_name+'/'+demo_name+'/vicon/'+obj
            vicon_files = sorted(glob.glob(vicon_dir+'/*.npy'))
            se3_dict[obj] = []
            
            for f in vicon_files:
                se3 = np.load(f)
                se3_dict[obj].append(np.expand_dims(se3,0))    
            se3_dict[obj] = np.concatenate(se3_dict[obj],0)
            data_len = se3_dict[obj].shape[0]
    
    output_dir = './data/'+task_name+'/'+demo_name+'/vicon_traj'
    util.create_dir(output_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']

    se3_objects = {}
    se3_traj = {}
    obj_color = {}
    for i, obj in enumerate(object_names):
        se3_objects[obj] = vision.SE3object(angle_type = 'axis')
        se3_traj[obj] = np.zeros([0,3])
        obj_color[obj] = colors[i]

    for t in range(data_len):
        ax.clear()
        for obj in object_names:
            se3_object = se3_objects[obj]
            
            se3 = se3_dict[obj][t,:]
            SE3 = se3_to_SE3(se3)
            T = SE3[0:3,3]
            se3_traj[obj] = np.concatenate([se3_traj[obj], np.expand_dims(T,0)],0)
            
            ax.plot(se3_traj[obj][:,0],se3_traj[obj][:,1],se3_traj[obj][:,2], 
                    obj_color[obj], alpha = 0.5, linewidth = 4)
            se3_object.apply_pose(se3)
            se3_object.plot(ax, scale = 0.05, linewidth = 3) 

        util.set_axes_equal(ax)
        #ax.axis('scaled')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')
        ax.view_init(elev = -60, azim = -90)
        fig.savefig(output_dir+'/%10d.png'%t)
    video_path = output_dir+'/object_trajectory.avi'
    util.frame_to_video(output_dir, video_path, 'png', fps=fps)()










