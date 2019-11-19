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
        
        args = parser.parse_args()
        task_name = args.task_name
        demo_name = args.demo_name
        VICON = args.vicon
    except:
        traceback.print_exc()
        print(colored('type like: python3 ./collect_data.py [task_name] [demo_name] [option]'),'r')
        print(colored('ex: python3 ./collect_data.py task1 demo0 --vicon'))
    
    if VICON:
        p_ros = subprocess.Popen('roslaunch babymind data_collect.launch', stdout = subprocess.PIPE, shell = True)
    else:
        p_ros = subprocess.Popen('roslaunch zed_wrapper zed.launch', stdout = None, shell = True)

    save_dir = './data/'+task_name+'/'+demo_name
    util.create_dir(save_dir, clear = True)
    while True:
        try:
            print('tuna')
            time.sleep(1)
        except:
            os.killpg(os.getpgid(p_ros.pid), signal.SIGTERM)
            


"""
try:
    
    p1 = subprocess.Popen('roslaunch zed_wrapper zed.launch', stdout = subprocess.PIPE, shell = True)
   
    #threading.Thread.
    #thread.start_new_thread(os.system, ('roslaunch zed_wrapper zed.launch',))
    #thread.start_new_thread(os.system, ('roslaunch vicon_bridge vicon.launch',))
    
    #os.system('roslaunch zed_wrapper zed.launch &')
    #os.system('roslaunch vicon_bridge vicon.launch')
except KeyboardInterrupt:
    pid = os.getpid()
    os.kill(pid)
    
"""