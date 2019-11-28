import glob
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import traceback
from termcolor import colored
import sys
import yaml

import IPython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from skimage import io
from skimage import transform

sys.path.append("./third_party/MASK_RCNN")
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images, apply_mask
import mrcnn.model as modellib
from mrcnn.model import log
from samples.coco import coco

from lib import util
from lib.image import *

def mask_to_label(mask, class_ids, class_names, using_class, wantShow = True):
    mask_h, mask_w, mask_ch = mask.shape
    class_ch = len(using_class)
    assert mask_ch == len(class_ids)

    label = np.zeros((mask_h, mask_w, class_ch))
    class_ids = class_ids
    reduced_origin_ids, reduced_new_ids = reduce_classids(class_ids, class_names, using_class)
    if not reduced_new_ids:
        return None
    else:
        for c, new_c in zip(reduced_origin_ids, reduced_new_ids):
            # insert mask pixel into image pixel when the pixel is not occupied
            where_mask = np.where(class_ids == c)
            
            if len(where_mask)>1:
                IPython.embed()
            for i in where_mask[0]:
                bool_idx = np.where(label[:,:,new_c]==0)
                label[:,:,new_c][bool_idx] = mask[:,:,i][bool_idx]

        return label

def reduce_classids(class_ids, class_names, using_class):
    '''
    reduced_origin_ids: indicies at using_class which are included in using class
    reduced_new_ids: indicies at class_names which are included in using_class
    '''
    reduced_new_ids = []
    reduced_origin_ids = []
    for c in class_ids:
        if class_names[c-1] in using_class:
            new_c = using_class.index(class_names[c-1])
            reduced_new_ids.append(new_c)
            reduced_origin_ids.append(c)
    return reduced_origin_ids, reduced_new_ids

def preprocess(config):   
    fps = config['animation']['fps']
    
    ## 1. resize image
    data_dir = './data/'+task_name
    demos = sorted(os.listdir(data_dir))
    for demo_name in demos:
        rgb_dir = data_dir +'/'+demo_name +'/rgb'
        depth_dir = data_dir+'/'+demo_name+'/depth'


    ## 2. generate mask from json
    task_name = config['task_name']
    object_file = './configure/'+task_name+'_objects.txt'
    class_names = util.load_txt(object_file)
    using_class = class_names

    demos = sorted(os.listdir('./output/'+task_name+'/label'))
    for demo_name in demos:
        json_path = './output/'+task_name+'/label/'+demo_name+'/annotations/segment_label.json'
        img_source_dir = './output/'+task_name+'/label/'+demo_name+'/labeled_img'
        if not os.path.exists(json_path):
            print('skip:' +str(demo_name))
            continue
        
        ## load json files from
        dataset = coco.CocoDataset()
        dataset.load_coco(json_path, img_source_dir)
        dataset.prepare()

        ## print dataset configuration
        print("Image Count: {}".format(len(dataset.image_ids)))
        print("Class Count: {}".format(dataset.num_classes))

        preprocess_dir = './output/'+task_name+'/preprocess/'+demo_name
        img_dir = preprocess_dir+'/labeled_image'
        mask_dir = preprocess_dir+'/mask'
        util.create_dir(img_dir,clear = True)
        util.create_dir(mask_dir,clear = True)
        
        count = 0
        for image_id in dataset.image_ids:
            
            image = dataset.load_image(image_id)
            mask, class_ids = dataset.load_mask(image_id)

            img_h, img_w, img_ch = image.shape
            '''
            ## Fit different size image into the same shaped
            image, window, scale, padding, _ = utils.resize_image(
                                                    image, 
                                                    min_dim=IMAGE_MIN_DIM, 
                                                    max_dim=IMAGE_MAX_DIM,
                                                    mode="square")
            mask = utils.resize_mask(mask, scale, padding)
            ## Resize image
            image = transform.resize(image, RESIZE_SHAPE, anti_aliasing = True)
            mask = transform.resize(mask, RESIZE_SHAPE, anti_aliasing = True)
            '''
            ## fit data for u-net
            label = mask_to_label(mask, class_ids, class_names, using_class)
            if label is None:
                pass
            else:
                label_img = multichannel_to_image(label)
                io.imsave(img_dir+'/%04d.png'%image_id, image)
                io.imsave(mask_dir+'/%04d.png'%image_id, label_img)
                np.save(img_dir+'/%04d.npy'%image_id, image)
                np.save(mask_dir+'/%04d.npy'%image_id, label)
                count += 1
    
