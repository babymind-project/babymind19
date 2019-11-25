import argparse
import os
import signal
import subprocess
import sys
from termcolor import colored
import time
import traceback

import cv2
import IPython
import pyzed.sl as sl

from lib import util

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('task_name')
        parser.add_argument('demo_name')
        parser.add_argument('--vicon', action = 'store_true')
        parser.add_argument('--watch', action = 'store_true')
        
        args = parser.parse_args()
        task_name = args.task_name
        demo_name = args.demo_name
        WATCH = args.watch
        VICON = args.vicon
    except:
        traceback.print_exc()
        print(colored('type like: python3 ./collect_data.py [task_name] [demo_name] [option]'),'r')
        print(colored('ex: python3 ./collect_data.py task1 demo0 --vicon'))
    
    ## Just watch camera output
    if WATCH:
        p_rviz = subprocess.Popen('roslaunch babymind image_visualize.launch', stdout = subprocess.PIPE, shell = True)
        input(colored('stop watch? [y/n]', 'green'))
        os.killpg(os.getpgid(p_rviz.pid), signal.SIGTERM)
        sys.exit()

    ## Log vision (+vicon) data
    object_names = util.load_txt('./configure/'+task_name+'_objects.txt')
    vicon_topics = ['/vicon/k_giant_ur3/k_giant_ur3', '/vicon/k_zed/k_zed']
    vicon_topics += ['/vicon/'+name+'/'+name for name in object_names ]
    camera_topics = ['/zed/zed_node/rgb/image_rect_color', '/zed/zed_node/depth/depth_registered']

    save_dir = './data/'+task_name+'/'+demo_name  
    if VICON:
        p_ros = subprocess.Popen('roslaunch babymind data_collect.launch', stdout = subprocess.PIPE, shell = True)
        topics = camera_topics + vicon_topics
    else:
        p_ros = subprocess.Popen('roslaunch babymind data_collect_no_vicon.launch', stdout =  subprocess.PIPE, shell = True)
        topics = camera_topics
    
    merged_topic = ''
    for topic in topics:
        merged_topic += (topic+' ')

    start_record = input(colored('Start record? [y/n]','green'))
    util.create_dir(save_dir, clear = True)  
    if start_record == 'y':
        print(colored('Press ctrl+c will finish record','green'))
        os.system('rosbag record -O %s/raw.bag %s'%(save_dir,merged_topic))
    print('terminate program')
    os.killpg(os.getpgid(p_ros.pid), signal.SIGTERM)
    sys.exit()