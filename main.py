from __future__ import division

import argparse
import datetime
import glob
import json
import os
import random
import sys
from termcolor import colored
import traceback
import yaml

import IPython
from lib import util

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('task_name')
        parser.add_argument('module_name')
        parser.add_argument('-c','--continuous', action = 'store_true')
        parser.add_argument('-t', '--test', action = 'store_true')
        parser.add_argument('-tr', '--train', action = 'store_true')

        args = parser.parse_args()
        task_name = args.task_name
        module_name = args.module_name
        CONTINUOUS = args.continuous
        TEST = args.test
        TRAIN = args.train
    except:
        traceback.print_exc()
        print('type like: python3 ./main.py [task_name] [module_name]')
    
    config = util.load_yaml('./configure/'+task_name+'.yaml')
    if module_name == 'preprocess' or module_name == 'pre':
        from algorithm.Preprocess import preprocess
        preprocess(config)

    elif module_name == 'segment' or module_name == 'seg':
        from algorithm.Segment_network import Segment_network
        seg_net = Segment_network(config)
        seg_net.build()
        if TEST:
            seg_net.test()
        else:
            seg_net.train(continuous = CONTINUOUS)
    
    elif module_name == 'pose':
        from algorithm.Pose_network import Pose_network
        pose_net = Pose_network(config, mode = 'default')
        if TEST:
            pose_net.batch_size = 1
            pose_net.build()
            pose_net.test()
        else:
            pose_net.build()
            pose_net.train(continuous=CONTINUOUS)

    elif module_name == 'read_pose':
        from algorithm.read_pose import read
        from algorithm.read_pose2 import read as read2
        '''
        read_pose :  draw 6d-pose in three-deimnsional space
        read_pose2 : draw 6d-pose projection on an image plane
        '''
        read(config)
        read2(config)

    elif module_name == 'compare_traj':
        from lib.module.compare_traj import compare
        compare(config)

    """    
    elif module_name == 'sfm_pose': # baseline1
        from lib.module.Sfm_pose_network import Sfm_pose_network
        pose_net = Sfm_pose_network(config, mode = 'default')
        if TEST:
            pose_net.batch_size = 1
            pose_net.build()
            pose_net.test()
        else:
            pose_net.build()
            pose_net.train(continuous=CONTINUOUS)
    
    elif module_name == 'se3_pose': # baseline2
        from lib.module.se3_pose_network import se3_pose_network
        pose_net = se3_pose_network(config, mode = 'default')
        if TEST:
            pose_net.batch_size = 1
            pose_net.build()
            pose_net.test()
        else:
            pose_net.build()
            pose_net.train(continuous=CONTINUOUS)

    elif module_name == 'read_pose':
        from lib.module.read_pose import read
        from lib.module.read_pose2 import read as read2
        '''
        read_pose :  draw 6d-pose in three-deimnsional space
        read_pose2 : draw 6d-pose projection on an image plane
        '''
        read(config)
        read2(config)


    elif module_name == 'imitation':
        from lib.module.Imitation import imitation
        pass

    elif module_name == 'ocisly':
        print('Of Course I Still Love You')

    """