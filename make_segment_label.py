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
import IPython
import pyzed.sl as sl

from lib import util

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('task_name')
        parser.add_argument('demo_name')

        args = parser.parse_args()
        task_name = args.task_name
        demo_name = args.demo_name
    except:
        traceback.print_exc()
        print(colored('type like: python3 ./make_segment_label.py [task_name] [demo_name]'),'r')
        print(colored('ex: python3 ./make_segment_label.py task1 demo0'))


    rgb_dir = './data/'+task_name+'/'+demo_name+'/rgb'
    rgb_imgs = sorted(glob.glob(rgb_dir+'/*.png'))
    class_file = './configure/'+task_name+'_objects.txt'
    label_dir = './output/'+task_name+'/label/'+demo_name+'/labeled_img'
    util.create_dir(label_dir, clear = True)
    json_file = './output/'+task_name+'/label/'+demo_name+'/segment_label'

    skip = input(colored('Pleas enter the number of skipping frames for eveery one label [current demo has %d images]'%len(rgb_imgs), 'green'))
    skip = int(skip)

    for i, rgb_img in enumerate(rgb_imgs):
        if i%skip == 0:
            rgb_file_name = os.path.basename(rgb_img)
            source_file = rgb_dir +'/'+rgb_file_name
            target_file = label_dir +'/'+rgb_file_name
            os.system('cp  %s %s'%(source_file, target_file))

    sys.path.append('./third_party/COCO_Style_Dataset_Generator_GUI')
    import third_party.COCO_Style_Dataset_Generator_GUI.segment as label_maker
    label_maker.run(image_dir = label_dir, class_file= class_file)

    import third_party.COCO_Style_Dataset_Generator_GUI.create_json_file as json_maker
    json_maker.run(image_dir = label_dir, class_file = class_file, file_type = 'png', file_path = json_file)

    IPython.embed()
