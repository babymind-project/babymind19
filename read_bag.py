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
        print(colored('type like: python3 ./read_bag.py [task_name] [option]'),'r')
        print(colored('ex: python3 ./read_bag.py task1 --clear'))
    
    task_dir = './data/'+task_name
    demo_dirs = sorted(os.listdir(task_dir))
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
        demo_dir = task_dir + '/' + demo_dir
        rgb_dir = demo_dir + '/rgb_raw'
        depth_dir = demo_dir + '/depth_raw'
        vicon_dir = demo_dir + '/vicon_raw'

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

                
        ## 3. extract rgb-images from long taken video and align depth and vicon data
        rgb_source_dir = demo_dir + '/rgb_raw'
        depth_source_dir = demo_dir + '/depth_raw'
        vicon_source_dir = demo_dir + '/vicon_raw'

        rgb_target_dir = demo_dir + '/rgb'
        depth_target_dir = demo_dir + '/depth'
        vicon_target_dir = demo_dir + '/vicon'

        util.create_dir(rgb_target_dir, clear = True)
        util.create_dir(depth_target_dir, clear = True)
        util.create_dir(vicon_target_dir, clear = True)
        print(colored('please check %s and inform us where to cut'%(demo_dir),'green') )
        #begin = input(colored('where to begin? [enter number as type of nanoseconds]'))
        #end = = input(colored('where to end? [enter number as type of nanoseconds]'))

        begin = 1574235759746797435
        end = 1574235760578664782
        ## extract rgb
        rgb_time = []
        for rgb_path in sorted(glob.glob(rgb_source_dir+'/*.npy')):
            file_name = os.path.basename(rgb_path)
            file_name_no_extension = os.path.splitext(file_name)[0]
            time_index = int(file_name_no_extension)

            if (time_index >= begin) and (time_index <=end):
                source_file = rgb_source_dir+'/'+file_name_no_extension+'.npy'
                target_file = rgb_target_dir+'/'+str(len(rgb_time)).zfill(10)+'.npy'
                os.system('cp  %s %s'%(source_file, target_file))

                source_file = rgb_source_dir+'/'+file_name_no_extension+'.png'
                target_file = rgb_target_dir+'/'+str(len(rgb_time)).zfill(10)+'.png'
                os.system('cp  %s %s'%(source_file, target_file))
                rgb_time.append( int(time_index) )

            if time_index >end:
                break

        ## align depth
        depth_files = sorted(glob.glob(depth_source_dir+'/*.npy'))
        indicator = 0
        depth_time = []
        for d1, d2 in zip(depth_files[:-1], depth_files[1:]):
            d1_name = os.path.basename(d1)
            d2_name = os.path.basename(d2)
            d1_no_ext = os.path.splitext(d1_name)[0]
            d2_no_ext = os.path.splitext(d2_name)[0]
            
            d1_time = int(d1_no_ext)
            d2_time = int(d2_no_ext)

            if d1_time <= rgb_time[indicator] <= d2_time:
                if rgb_time[indicator] - d1_time <= d2_time- rgb_time[indicator]:
                    copy_file_no_ext = d1_no_ext
                else:
                    copy_file_no_ext =  d2_no_ext

                source_file = depth_source_dir+'/'+copy_file_no_ext+'.npy'
                target_file = depth_target_dir+'/'+str(indicator).zfill(10)+'.npy'
                os.system('cp  %s %s'%(source_file, target_file))

                source_file = depth_source_dir+'/'+copy_file_no_ext+'.png'
                target_file = depth_target_dir+'/'+str(indicator).zfill(10)+'.png'
                os.system('cp  %s %s'%(source_file, target_file))
                indicator += 1
                depth_time.append(int(copy_file_no_ext))
            else:
                pass

            if indicator >= len(rgb_time):
                break

        ## align vicon
        vicon_files = sorted(glob.glob(vicon_source_dir+'/*.npy'))
        indicator = 0
        vicon_time = []
        for v1, v2 in zip(vicon_files[:-1], vicon_files[1:]):
            v1_name = os.path.basename(V1)
            v2_name = os.path.basename(v2)
            v1_no_ext = os.path.splitext(v1_name)[0]
            v2_no_ext = os.path.splitext(v2_name)[0]
            
            v1_time = int(v1_no_ext)
            v2_time = int(v2_no_ext)

            if v1_time <= rgb_time[indicator] <= v2_time:
                if rgb_time[indicator] - v1_time <= v2_time- rgb_time[indicator]:
                    copy_file_no_ext = v1_no_ext
                else:
                    copy_file_no_ext =  v2_no_ext

                source_file = vicon_source_dir+'/'+copy_file_no_ext+'.npy'
                target_file = vicon_target_dir+'/'+str(indicator).zfill(10)+'.npy'
                os.system('cp  %s %s'%(source_file, target_file))

                indicator += 1
                vicon_time.append(int(copy_file_no_ext))
            else:
                pass

            if indicator >= len(rgb_time):
                break


        IPython.embed()
           
            






