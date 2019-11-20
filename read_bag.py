import argparse
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
import numpy as np
import rosbag

from lib import util

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
        print(colored('type like: python3 ./collect_data.py [task_name] [option]'),'r')
        print(colored('ex: python3 ./collect_data.py task1 --clear'))
    
    task_dir = './data/'+task_name
    demo_dirs = os.listdir(task_dir)
    for demo_dir in demo_dirs:
        ## 1. reindex rosbag file
        reindex_dir = task_dir + '/' + demo_dir +'/reindex'
        bag_path = task_dir + '/' + demo_dir +'/raw.bag.active' 
        util.create_dir(reindex_dir, clear = True)
        
        print(colored('reindex bag file','blue'))
        reindex_cmd = "rosbag reindex --output-dir=" + reindex_dir +" "+bag_path
        os.system(reindex_cmd)

        ## 2. extract infomation from reindexed bag
        print(colored('extract bag file','blue'))
        rgb_dir = task_dir + '/' + demo_dir + '/rgb'
        depth_dir = task_dir + '/' + demo_dir + '/depth'
        vicon_dir = task_dir + '/' + demo_dir + '/vicon'

        util.create_dir(rgb_dir, clear = True)
        util.create_dir(depth_dir, clear = True)
        util.create_dir(vicon_dir, clear = True)
        
        reindex_path = reindex_dir +'/raw.bag.active'
        bag = rosbag.Bag(reindex_path)
        bridge = CvBridge()
        for topic, msg, t in bag.read_messages():
            ros_time = msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs
        
            if topic == '/zed/zed_node/rgb/image_rect_color':
                rgb_file = rgb_dir + '/' + str(t)
                cv_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(rgb_file+'.png', cv_rgb)
                np.save(rgb_file+'.npy', cv_rgb)
                # cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2GRAY)

            elif topic == '/zed/zed_node/depth/depth_registered':
                depth_file = depth_dir + '/' + str(t)
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                cv2.imwrite(depth_file+'.png', cv_depth)
                np.save(depth_file+'.npy', cv_depth)
                
            elif topic == '/vicon/k_sweeper/k_sweeper':
                vicon_file = vicon_dir + '/' + str(t)
                np.save(vicon_file+'.npy', msg)

                
                