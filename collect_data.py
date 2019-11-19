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
    
    if WATCH:
        p_rviz = subprocess.Popen('roslaunch babymind image_visualize.launch', stdout = subprocess.PIPE, shell = True)
        input(colored('stop watch? [y/n]', 'green'))
        os.killpg(os.getpgid(p_rviz.pid), signal.SIGTERM)
        sys.exit()
        
    if VICON:
        p_ros = subprocess.Popen('roslaunch babymind data_collect.launch', stdout = subprocess.PIPE, shell = True)
    else:
        p_ros = subprocess.Popen('roslaunch babymind data_collect_no_vicon.launch', stdout = None, shell = True)

    save_dir = './data/'+task_name+'/'+demo_name
    util.create_dir(save_dir, clear = True)
    
    start_record = input(colored('start record? [y/n]','green'))
    if start_record == 'y':
        print(colored('ctrl+c will finish record','green'))
        os.system('rosbag record -O '+save_dir+'/raw.bag'+' /zed/zed_node/stereo/image_rect_color /vicon/k_sweeper/k_sweeper')
    print('terminate program')
    os.killpg(os.getpgid(p_ros.pid), signal.SIGTERM)
    sys.exit()