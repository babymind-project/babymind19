import datetime
import glob
import os
import time
import traceback

import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from termcolor import colored
import tflearn
from tflearn.layers.core import fully_connected, flatten
from tflearn.layers.conv import conv_2d, conv_2d_transpose
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization

from lib import network
from lib import util
from lib import vision
from lib.image import *
from lib.math import *
from lib.preprocess import *
from lib.learner import *

class Pose_stream(Batch_stream):
    def __init__(self, task_name, batch_size, scale = 1, mode = 'train', supervision = 'never'):
        self.batch_size = batch_size
        self.mode = mode
        self.supervision = supervision  # 'never', 'full', both_ends', ...
        print('supervision'+colored(self.supervision,'blue'))
        self.img_dir = './data'+'/'+task_name
        self.mask_dir = './output/segment/' + task_name
        self.preprocess_dir = './preprocess/'+task_name
        self.vicon_dir = './output/vicon/'+task_name

        self.num_data = 0
        self._count = 0

        self.demo_list = os.listdir(self.img_dir)
        self.img_dict = {}
        self.depth_dict = {}
        self.mask_dict = {}
        self.vicon_dict = {}
        self.g_vr_dict = {}

        self.img_preprocess_dict = {}
        self.depth_preprocess_dict = {}
        self.nan_mask_preprocess_dict = {}
        self.mask_preprocess_dict = {}
        self.bbox_preprocess_dict = {}
        self.vicon_preprocess_dict = {}
        self.usingVicon_preprocess_dict = {}
        self.g_vr_preprocess_dict = {}

        for demo in self.demo_list:
            img_demo_dir = self.img_dir+'/'+demo
            mask_demo_dir = self.mask_dir +'/'+demo
            vicon_demo_dir = self.vicon_dir+'/'+demo
            self.img_dict[demo] = sorted(glob.glob(img_demo_dir+'/image/*.npy'))
            self.depth_dict[demo] = sorted(glob.glob(img_demo_dir+'/depth/*.npy'))
            self.mask_dict[demo] = sorted(glob.glob(mask_demo_dir+'/*.npy'))
            self.vicon_dict[demo] = vicon_demo_dir+'/aligned.npy'
            self.g_vr_dict[demo] = img_demo_dir+'/camera_position0.npy' # vicon to robot coordinates

        for demo in self.demo_list:
            img_list = self.img_dict[demo]
            depth_list = self.depth_dict[demo]
            mask_list = self.mask_dict[demo]
            vicon_list = self.vicon_dict[demo]
            g_vr_list = self.g_vr_dict[demo]

            preprocess_dir = self.preprocess_dir+'/'+demo
            preprocess_img_dir = preprocess_dir +'/image'
            preprocess_depth_dir = preprocess_dir +'/depth'
            preprocess_nan_mask_dir = preprocess_dir +'/nan_mask'
            preprocess_mask_dir = preprocess_dir +'/mask'
            preprocess_mask_img_dir = preprocess_dir +'/mask_img'
            preprocess_bbox_dir = preprocess_dir +'/bbox'
            preprocess_vicon_dir = preprocess_dir+'/vicon'
            preprocess_usingVicon_dir = preprocess_dir +'/usingVicon'
            preprocess_g_vr_dir = preprocess_dir + '/g_vr'


            util.create_dir(preprocess_img_dir, clear = True)
            util.create_dir(preprocess_depth_dir, clear = True)
            util.create_dir(preprocess_nan_mask_dir, clear = True)
            util.create_dir(preprocess_mask_dir, clear = True) 
            util.create_dir(preprocess_mask_img_dir, clear = True) 
            util.create_dir(preprocess_bbox_dir, clear = True) 
            util.create_dir(preprocess_vicon_dir, clear = True)
            util.create_dir(preprocess_usingVicon_dir, clear = True)
            util.create_dir(preprocess_g_vr_dir, clear = True)

            vicon_load = np.load(vicon_list)
            g_vr_load = np.load(g_vr_list, allow_pickle = True)
            g_vr_quater = g_vr_load[1]+g_vr_load[2] # append list
            g_vr_se3 = quaternion_to_se3(g_vr_quater) 
            g_vr_SE3 = se3_to_SE3(g_vr_se3)  # rescale
            
            index = 0
            total_len = len(img_list)
            for img_path, depth_path, mask_path in zip(img_list, depth_list, mask_list):
                img_load = np.load(img_path)
                depth_load = np.load(depth_path)
                mask_load = np.load(mask_path)
                
                img = preprocess_img(img_load, scale=scale)
                depth, nan_mask= preprocess_depth(depth_load[:,:,2], scale=scale)
                mask = preprocess_mask(mask_load, scale=scale)
                bbox = preprocess_bbox(depth_load, mask_load, scale = scale)
                g_vr = np.copy(g_vr_SE3)

                if self.supervision == 'full':
                    usingVicon = [1]
                    vicon = vicon_load[index,:,:]
                elif self.supervision == 'never':
                    usingVicon = [0]
                    vicon = vicon_load[0,:,:]
                elif self.supervision == 'both_ends': 
                    if (index == 0) or (index == total_len-1):
                        usingVicon = [1]
                        vicon = vicon_load[index,:,:]
                    else:
                        usingVicon = [0]
                        vicon = vicon_load[index,:,:]
                elif self.supervision == 'one_end':
                    if index == 0:
                        usingVicon = [1]
                        vicon = vicon_load[index,:,:]
                    else:
                        usingVicon = [0]
                        vicon = vicon_load[index,:,:]
                else:
                    raise NotImplementedError
                
                img = np.expand_dims(img, axis = 0)
                depth = np.expand_dims(depth, axis = 0)
                mask = np.expand_dims(mask, axis = 0)
                nan_mask = np.expand_dims(nan_mask, axis = 0)
                bbox = np.expand_dims(bbox, axis = 0)
                vicon = np.expand_dims(vicon, 0)
                usingVicon = np.expand_dims(usingVicon, 0)
                g_vr = np.expand_dims(g_vr,0)

                np.save(preprocess_img_dir+'/%04d.npy'%index, img)
                np.save(preprocess_depth_dir+'/%04d.npy'%index, depth)
                np.save(preprocess_nan_mask_dir+'/%04d.npy'%index, nan_mask)
                np.save(preprocess_mask_dir+'/%04d.npy'%index, mask)
                np.save(preprocess_bbox_dir+'/%04d.npy'%index, bbox)
                np.save(preprocess_vicon_dir+'/%04d.npy'%index, vicon)
                np.save(preprocess_usingVicon_dir+'/%04d.npy'%index, usingVicon)
                np.save(preprocess_g_vr_dir+'/%04d.npy'%index, g_vr)

                mask_img = multichannel_to_image(mask[0,:])
                np.save(preprocess_mask_img_dir+'/%04d.png'%index, mask_img)

                index+=1
            
            self.img_preprocess_dict[demo] = sorted(glob.glob(preprocess_img_dir+'/*.npy'))
            self.depth_preprocess_dict[demo] = sorted(glob.glob(preprocess_depth_dir+'/*.npy'))
            self.nan_mask_preprocess_dict[demo] = sorted(glob.glob(preprocess_nan_mask_dir+'/*.npy'))
            self.mask_preprocess_dict[demo] = sorted(glob.glob(preprocess_mask_dir+'/*.npy'))
            self.bbox_preprocess_dict[demo] = sorted(glob.glob(preprocess_bbox_dir+'/*.npy'))
            self.vicon_preprocess_dict[demo] = sorted(glob.glob(preprocess_vicon_dir+'/*.npy'))
            self.usingVicon_preprocess_dict[demo] = sorted(glob.glob(preprocess_usingVicon_dir+'/*.npy'))
            self.g_vr_preprocess_dict[demo] = sorted(glob.glob(preprocess_g_vr_dir+'/*.npy'))
            self.num_data += len(self.img_dict[demo])
        
    def iterator(self, batch_size):
        self._count = 0

        for demo in self.demo_list:
            img_list = self.img_preprocess_dict[demo]
            depth_list = self.depth_preprocess_dict[demo]
            nan_mask_list = self.nan_mask_preprocess_dict[demo]
            mask_list = self.mask_preprocess_dict[demo]
            bbox_list = self.bbox_preprocess_dict[demo]
            vicon_list = self.vicon_preprocess_dict[demo]
            usingVicon_list = self.usingVicon_preprocess_dict[demo]
            g_vr_list = self.g_vr_preprocess_dict[demo]

            img0_list = img_list[0:-2]
            img1_list = img_list[1:-1]
            depth0_list = depth_list[0:-2]
            depth1_list = depth_list[1:-1]
            nan_mask0_list = nan_mask_list[0:-2]
            nan_mask1_list = nan_mask_list[1:-1]
            mask0_list = mask_list[0:-2]
            mask1_list = mask_list[1:-1]
            bbox0_list = bbox_list[0:-2]
            bbox1_list = bbox_list[1:-1]
            vicon0_list = vicon_list[0:-2]
            vicon1_list = vicon_list[1:-1]
            vicon2_list = vicon_list[2:]
            usingVicon0_list = usingVicon_list[0:-2]
            usingVicon1_list = usingVicon_list[1:-1]
            usingVicon2_list = usingVicon_list[2:]
            g_vr0_list = g_vr_list[0:-2]
            g_vr1_list = g_vr_list[1:-1] 
            g_vr2_list = g_vr_list[2:]
            data_len = len(img0_list)
                
            idx = np.arange(data_len)
            if self.mode == 'train':
                random.shuffle(idx)
            elif self.mode == 'test':
                pass

            for i in idx:
                img0_path = img0_list[i]
                img1_path = img1_list[i]
                depth0_path = depth0_list[i]
                depth1_path = depth1_list[i]
                nan_mask0_path = nan_mask0_list[i]
                nan_mask1_path = nan_mask1_list[i]
                mask0_path = mask0_list[i]
                mask1_path = mask1_list[i]
                bbox0_path = bbox0_list[i]
                bbox1_path = bbox1_list[i]
                vicon0_path = vicon0_list[i]
                vicon1_path = vicon1_list[i]
                vicon2_path = vicon2_list[i]
                usingVicon0_path = usingVicon0_list[i]
                usingVicon1_path = usingVicon1_list[i]
                usingVicon2_path = usingVicon2_list[i]
                g_vr0_path = g_vr0_list[i]
                g_vr1_path = g_vr1_list[i]
                g_vr2_path = g_vr2_list[i]

                img0 = np.load(img0_path)
                img1 = np.load(img1_path)
                depth0 = np.load(depth0_path)
                depth1 = np.load(depth1_path)
                nan_mask0 = np.load(nan_mask0_path)
                nan_mask1 = np.load(nan_mask1_path)
                mask0 = np.load(mask0_path)
                mask1 = np.load(mask1_path)
                bbox0 = np.load(bbox0_path)
                bbox1 = np.load(bbox1_path)
                vicon0 = np.load(vicon0_path)
                vicon1 = np.load(vicon1_path)
                usingVicon0 = np.load(usingVicon0_path)
                usingVicon1 = np.load(usingVicon1_path)
                g_vr0 = np.load(g_vr0_path)
                g_vr1 = np.load(g_vr1_path)
                self._count +=1
                yield img0, img1, depth0, depth1, \
                        mask0, mask1, nan_mask0, nan_mask1, \
                        vicon0, vicon1, usingVicon0, usingVicon1, g_vr0, g_vr1, \
                        bbox0, bbox1 , demo
            
class se3_pose_network(Network):
    def __init__(self, config, mode='default'):
        self.tensors =[]
        self.operations =[]
        self.graph = []
        self.sess = []
        self.saver = []
        self.init = []

        self.mode = mode
        self.network_name = 'pose'
        self.task_name = config['data_name']
        self.supervision = config['pose']['supervision']
        self.scale = config['pose']['scale']
        self.batch_size = config['pose']['batch_size']  
        self.learning_rate = config['pose']['learning_rate']

        self.depth_max = config['camera']['depth_max']
        self.depth_min = config['camera']['depth_min']
        ###
        self.img_size =  config['preprocess']['image_size']
        self.img_size = [int(i*self.scale) for i in self.img_size]
        self.mask_ch = config['object']['nb_object']

        self.weight_dir = './weight/'+self.network_name+'/'+self.task_name
        self.figure_dir = './figure/'+self.network_name+'/'+self.task_name
        self.output_dir = './output/'+self.network_name+'/'+self.task_name
        self._error = 0

    def normalize_depth(self, depth):
        MAX = self.depth_max/1000 # (m to mm)
        MIN = self.depth_min/1000 
        normalized_depth = (depth-MIN)/(MAX-MIN)
        #normalized_depth = np.clip(normalized_depth, 0, 1)
        return normalized_depth

    def unnormalize_depth(self, normalized_depth):
        # default MIN&MAX value is from zed-mini
        MAX = self.depth_max/1000 # (m to mm)
        MIN = self.depth_min/1000
        depth = (MAX-MIN)*normalized_depth+MIN
        return depth 

    def get_photometric_loss(self, frame0, frame1_warped, mask, nan_mask):
        h = self.img_size[0]
        w = self.img_size[1]
        ch = self.mask_ch

        diff = tf.reduce_sum(tf.square(frame0[:,:-1,:-1,:]-frame1_warped[:,:-1,:-1,:]),3)
        loss = tf.reduce_mean(diff)
        return loss
    
    def get_pc_loss(self, pc0, motion_map, transformed_pc1, mask, depth_only = True):
        h = self.img_size[0]
        w = self.img_size[1]
        ch = self.mask_ch
        CLIP = True

        motion = tf.reshape(motion_map, [-1,ch+1,3,h,w]) 
        motion = tf.transpose(motion, [0, 1, 3, 4, 2])
        # assumption : motion mask is not overlapped
        motion = tf.reduce_sum(motion, axis = 1) 
        transformed_pc0= tf.add(pc0, motion)

        norm_pc0 = self.normalize_depth(transformed_pc0[:,:,:,2])
        norm_pc1 = self.normalize_depth(transformed_pc1[:,:,:,2])
        
        if CLIP:
            rate = tf.constant(0.9)
            for i in range(self.batch_size):
                diff_abs = tf.reshape(tf.square(norm_pc0[i,:,:] - norm_pc1[i,:,:]), [h,w])
                masked_diff_abs = tf.gather_nd(diff_abs, tf.where(mask[i,:,:,0]>0.1))
                size = tf.shape(masked_diff_abs)[0]
                masked_diff_abs = tf.reshape(masked_diff_abs, [1, size])

                bottom_k_value, _ = tf.nn.top_k(-masked_diff_abs, tf.to_int32( tf.to_float(size)*rate ))
                bottom_k_value = -bottom_k_value
                if i == 0:
                    loss = tf.reduce_mean(bottom_k_value)
                else:
                    loss += tf.reduce_mean(bottom_k_value)
        else:
            loss = tf.reduce_mean(tf.square(norm_pc0-norm_pc1), axis = 0)
        loss = tf.reduce_mean(loss)
        return loss, transformed_pc0

    def get_RGB_recover_loss(self, sensor_rgb, recover_rgb):
        return tf.reduce_mean(tf.square(sensor_rgb-recover_rgb))
    
    def get_depth_recover_loss(self, sensor_depth, recovered_depth, nan_mask):
        normalized_sensor = self.normalize_depth(sensor_depth)
        normalized_recover = self.normalize_depth(recovered_depth)
        
        diff = tf.square(tf.subtract(normalized_sensor, normalized_recover))
        loss = tf.multiply(1-nan_mask, diff)
        loss = tf.reduce_mean(loss)
        return loss

    def get_volume_loss(self, bbox, se3):
        volume_loss = tf.zeros((self.batch_size,))
        x_pm = 5e-2 # x-plus margin
        x_mm = 5e-2 # x-minus margin
        y_pm = 5e-2
        y_mm = 5e-2
        z_pm = 5e-2
        z_mm = 1e-1
        for i in range(self.batch_size):
            for k in range(self.mask_ch):
                SE3 = tf_se3_to_SE3(se3[:,k,:])
                maxx = bbox[i, k, 0:3]
                minn = bbox[i, k, 3:6]
                volume_loss += tf_minmax_quadratic( SE3[i,0,3], minn[0]-x_mm, maxx[0]+x_pm)
                volume_loss += tf_minmax_quadratic( SE3[i,1,3], minn[1]-x_mm, maxx[1]+x_pm)
                volume_loss += tf_minmax_quadratic( SE3[i,2,3], minn[2]-x_mm, maxx[2]+x_pm)
        return tf.reduce_mean(volume_loss)

    def get_G(self, se3_0, se3_1):
        '''
        SE3_1^-1 = [G_SE3] * SE3_0^-1
        '''
        mask_ch = self.mask_ch
        G_SE3 = []
        for m in range(mask_ch):
            SE3_0 = tf_se3_to_SE3(se3_0[:,m,:] )
            SE3_1 = tf_se3_to_SE3(se3_1[:,m,:] )
            SE3_0_inv = tf_inv_SE3(SE3_0)

            G_SE3_obj = tf.matmul(SE3_1, SE3_0_inv)
            G_SE3.append(tf.expand_dims(G_SE3_obj,axis = 1))
        G_SE3 = tf.concat(G_SE3, axis = 1)
        return G_SE3

    def build(self):
        batch_size = self.batch_size
        mode = self.mode
        img_size = self.img_size
        mask_ch = self.mask_ch
        learning_rate = self.learning_rate
        scale = self.scale
        hope = 1e-10 # epsilon value for preventing machine zero

        self.sess = tf.Session()
        with tf.variable_scope('pose'):
            frame0_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], 3))
            frame1_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], 3))
            depth0_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1]))
            depth1_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1]))
            mask0_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], mask_ch))
            mask1_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], mask_ch))
            nan0_ph = tf.placeholder(dtype = tf.float32, shape = (None, img_size[0], img_size[1]))
            nan1_ph = tf.placeholder(dtype = tf.float32, shape = (None, img_size[0], img_size[1]))
            bbox0_ph = tf.placeholder(dtype = tf.float32, shape = (None,mask_ch, 6))
            bbox1_ph = tf.placeholder(dtype = tf.float32, shape = (None,mask_ch, 6))
            vicon0_ph = tf.placeholder(dtype = tf.float32, shape = (None, mask_ch, 6))
            vicon1_ph = tf.placeholder(dtype = tf.float32, shape = (None, mask_ch, 6))
            usingVicon0_ph = tf.placeholder(dtype = tf.float32, shape = (None, 1))
            usingVicon1_ph = tf.placeholder(dtype = tf.float32, shape = (None, 1))
            g_vr0_ph = tf.placeholder(dtype = tf.float32, shape = (None, 4, 4))
            g_vr1_ph = tf.placeholder(dtype = tf.float32, shape = (None, 4, 4))

            # used vision module
            cloud_transformer = vision.Cloud_transformer(intrinsic = 'zed_mini', scale = scale)    
            optical_transformer = vision.Optical_transformer(intrinsic='zed_mini', scale = scale, mask_ch = mask_ch+1, input_type ='SO3')
            image_warper_backward = vision.Image_warper_backward()

            # sensor depth to full depth
            depth0_normalized_ = self.normalize_depth(depth0_ph)
            depth0_rgb_concat_ = tf.concat([tf.expand_dims(depth0_normalized_, axis = 3), frame0_ph], axis = 3)

            depth1_normalized_ = self.normalize_depth(depth1_ph)
            depth1_rgb_concat_ = tf.concat([tf.expand_dims(depth1_normalized_, axis = 3), frame1_ph], axis = 3)
            
            if self.mode == 'default':
                unet0_logit_, unet0_sigmoid_, embed0_ = network.u_net(depth0_rgb_concat_, output_ch = 4, trainable = True)
                unet1_logit_, unet1_sigmoid_, embed1_ = network.u_net(depth1_rgb_concat_, output_ch = 4, reuse = True, trainable = True)
            elif self.mode == 'fine':
                unet0_logit_, unet0_sigmoid_, embed0_ = network.u_net(depth0_rgb_concat_, output_ch = 4, trainable = False)
                unet1_logit_, unet1_sigmoid_, embed1_ = network.u_net(depth1_rgb_concat_, output_ch = 4, reuse = True, trainable = False)
            
            rgb0_sigmoid_ = unet0_sigmoid_[:,:,:,1:4]
            rgb1_sigmoid_ = unet1_sigmoid_[:,:,:,1:4]
            depth0_sigmoid_ = tf.expand_dims(unet0_sigmoid_[:,:,:,0],3)
            depth1_sigmoid_ = tf.expand_dims(unet1_sigmoid_[:,:,:,0],3)
            recovered_depth0_ = self.unnormalize_depth(depth0_sigmoid_)[:,:,:,0] 
            recovered_depth1_ = self.unnormalize_depth(depth1_sigmoid_)[:,:,:,0] 
            full_depth0_ = tf.multiply(nan0_ph, recovered_depth0_) + tf.multiply(1-nan0_ph, depth0_ph)
            full_depth1_ = tf.multiply(nan1_ph, recovered_depth1_) + tf.multiply(1-nan1_ph, depth1_ph)

            ## add background
            mask0_ = tf.concat( [tf.zeros_like(tf.expand_dims(mask0_ph[:,:,:,0],3)), mask0_ph], 3)
            mask1_ = tf.concat([tf.zeros_like(tf.expand_dims(mask1_ph[:,:,:,0],3)), mask1_ph],3)
            
            # parametrized rotation & translation
            embed0_ = tf.layers.flatten(embed0_)
            embed1_ = tf.layers.flatten(embed1_)

            # fully connected layer
            se3_0_ = network.pose_net(embed0_, mask_ch, reuse = False) # [N,K,6]
            se3_1_ = network.pose_net(embed1_, mask_ch, reuse = True) # [N,K,6]

            usingVicon0_ = tf.expand_dims(usingVicon0_ph, 1)
            usingVicon1_ = tf.expand_dims(usingVicon1_ph, 1)
            se3_0_hard_ = tf.multiply(usingVicon0_, vicon0_ph)+tf.multiply(1-usingVicon0_, se3_0_)
            se3_1_hard_ = tf.multiply(usingVicon1_, vicon1_ph)+tf.multiply(1-usingVicon1_, se3_1_)


            G_SE3_ = self.get_G(se3_0_ , se3_1_)
            if self.supervision == 'never':                
                # actual notation : 
                # g_vr * g_rc = g_vc
                # g_vc * g_co1 = g_vo1
                # g_c2c1 = g_vc^-1 * g_vo2 * g_vo1^-1 * g_cv^-1
                se3_rc_ = tf.Variable( np.asarray([[0.38457832, 0.09596953, -0.32798763, -0.34428167, 0.43178049, -0.17598158]],dtype = np.float32), trainable = True) # camera to vicon
                se3_rc_ = tf.tile(se3_rc_, [batch_size, 1])
                g_rc_ = tf_se3_to_SE3(se3_rc_)
                g_vc1_ = tf.matmul(g_vr0_ph, g_rc_)
                g_vc2_ = tf.matmul(g_vr1_ph, g_rc_)  
                g_vc1_ = tf.reshape(g_vc1_, [-1,1,4,4]) # assert object num = 1
                g_vc2_ = tf.reshape(g_vc2_, [-1,1,4,4])        
                
            elif (self.supervision == 'both_ends') or (self.supervision == 'full'):                
                # actual notation : 
                # g_vr * g_rc = g_vc
                # g_vc * g_co1 = g_vo1
                # g_c2c1 = g_vc^-1 * g_vo2 * g_vo1^-1 * g_vc
                se3_rc_ = tf.Variable( np.asarray([[0.38457832, 0.09596953, -0.32798763, -0.34428167, 0.43178049, -0.17598158]],dtype = np.float32), trainable = True) # camera to vicon
                se3_rc_ = tf.tile(se3_rc_, [batch_size, 1])

                g_rc_ = tf_se3_to_SE3(se3_rc_)
                g_vc1_ = tf.matmul(g_vr0_ph, g_rc_)
                g_vc2_ = tf.matmul(g_vr1_ph, g_rc_)  
                g_vc1_ = tf.reshape(g_vc1_, [-1,1,4,4]) # assert object num = 1
                g_vc2_ = tf.reshape(g_vc2_, [-1,1,4,4])       
                g_vc2_inv_ = tf.reshape(tf_inv_SE3(tf.reshape(g_vc2_,[-1,4,4])),[-1,mask_ch,4,4])
                G_SE3_ = tf.matmul(tf.matmul(g_vc2_inv_,G_SE3_),g_vc1_)

            G_SE3_inv_ = tf.reshape(tf_inv_SE3(tf.reshape(G_SE3_,[-1,4,4])),[-1, mask_ch,4,4])
            
            f_obj_t_ = G_SE3_[:,:,0:3,3]
            f_obj_SO3_ = G_SE3_[:,:,0:3,0:3]
            b_obj_t_ = G_SE3_inv_[:,:,0:3,3]
            b_obj_SO3_ = G_SE3_inv_[:,:,0:3,0:3]

            obj_t_zero_ = tf.reduce_sum(tf.zeros_like(f_obj_t_),1,keepdims=True)
            obj_SO3_zero_ = tf.reduce_sum(tf.zeros_like(f_obj_SO3_),1,keepdims=True)
            f_obj_motion_ = (tf.concat([obj_t_zero_, f_obj_t_],1), tf.concat([obj_SO3_zero_,f_obj_SO3_],1), mask0_)
            b_obj_motion_ = (tf.concat([obj_t_zero_, b_obj_t_],1), tf.concat([obj_SO3_zero_,b_obj_SO3_],1), mask1_)
            f_cam_motion_ = (tf.zeros_like(f_obj_t_[:,0,:]), tf.zeros_like(f_obj_SO3_[:,0,:]))
            b_cam_motion_ = (tf.zeros_like(b_obj_t_[:,0,:]), tf.zeros_like(b_obj_SO3_[:,0,:]))
            
            # vision transformation
            point_cloud0_ = cloud_transformer(full_depth0_)
            point_cloud1_ = cloud_transformer(full_depth1_)
            
            pc0_ = tf.reshape(point_cloud0_, [-1,3,img_size[0],img_size[1] ] )
            pc0_ = tf.transpose( pc0_, [0,2,3,1])
            pc1_ = tf.reshape(point_cloud1_, [-1,3,img_size[0],img_size[1] ] )
            pc1_ = tf.transpose( pc1_, [0,2,3,1])

            f_pix_pos_, f_flow_, f_point_cloud1_, f_motion_map_ = optical_transformer(point_cloud0_, f_cam_motion_, f_obj_motion_)
            b_pix_pos_, b_flow_, b_point_cloud0_, b_motion_map_ = optical_transformer(point_cloud1_, b_cam_motion_, b_obj_motion_)
            
            frame1_warped_ = image_warper_backward(frame1_ph, f_pix_pos_)
            pc1_warped_ = image_warper_backward(pc1_, f_pix_pos_)

            frame0_warped_ = image_warper_backward(frame0_ph, b_pix_pos_)
            pc0_warped_ = image_warper_backward(pc0_, b_pix_pos_)

            vicon_supervision_loss0_ = tf.reduce_sum(tf.square( tf_se3_to_SE3(vicon0_ph[:,0,:])-tf_se3_to_SE3(se3_0_[:,0,:])),[1,2])
            vicon_supervision_loss0_ = tf.multiply(usingVicon0_ph,vicon_supervision_loss0_)
            vicon_supervision_loss0_ = tf.reduce_mean(vicon_supervision_loss0_)

            vicon_supervision_loss1_ = tf.reduce_sum(tf.square( tf_se3_to_SE3(vicon1_ph[:,0,:])-tf_se3_to_SE3(se3_1_[:,0,:])),[1,2])
            vicon_supervision_loss1_ = tf.multiply(usingVicon1_ph, vicon_supervision_loss1_)
            vicon_supervision_loss1_ = tf.reduce_mean(vicon_supervision_loss1_)

            ## losses
            # (1) photometric loss
            f_photometric_loss_ = self.get_photometric_loss(frame0_ph, frame1_warped_, mask0_[:,:,:,1:], nan0_ph)
            b_photometric_loss_ = self.get_photometric_loss(frame1_ph, frame0_warped_, mask1_[:,:,:,1:], nan1_ph)
            # (2) pc loss
            f_pc_loss_ , motioned_pc0_ = self.get_pc_loss(pc0_,f_motion_map_, pc1_warped_, mask0_[:,:,:,1:], depth_only = True)
            b_pc_loss_, motioned_pc1_ = self.get_pc_loss(pc1_,b_motion_map_, pc0_warped_, mask1_[:,:,:,1:], depth_only = True)
            # (3) rgb recon loss
            rgb_recover_loss0_ = self.get_RGB_recover_loss(rgb0_sigmoid_,frame0_ph)
            rgb_recover_loss1_ = self.get_RGB_recover_loss(rgb1_sigmoid_,frame1_ph)
            # (4) dept recover loss
            depth_recover_loss0_ = self.get_depth_recover_loss(depth0_ph, recovered_depth0_, nan0_ph)
            depth_recover_loss1_ = self.get_depth_recover_loss(depth1_ph, recovered_depth1_, nan1_ph)  
            # (5) volume loss     
            volume_loss0_ = self.get_volume_loss(bbox0_ph, se3_0_)
            volume_loss1_ = self.get_volume_loss(bbox1_ph, se3_1_)

            ## sum loss
            photometric_loss_ = f_photometric_loss_+b_photometric_loss_
            pc_loss_ = f_pc_loss_ + b_pc_loss_   
            rgb_recover_loss_ = rgb_recover_loss0_+rgb_recover_loss1_
            depth_recover_loss_ = depth_recover_loss0_ + depth_recover_loss1_   
            volume_loss_ = volume_loss0_ + volume_loss1_
            vicon_supervision_loss_ = vicon_supervision_loss0_ + vicon_supervision_loss1_

            if self.supervision =='full':
                self._p = 1e-20 
                self._pc = 0 
                self._d = 1e-2  
                self._recon = 1e-1
                self._vc = 1e0
                self._v = 0
            
            elif self.supervision == 'both_ends':
                self._p = 1e-20
                self._pc = 1e-1 
                self._d = 1e-2  
                self._recon = 1e-2 
                self._vc = 1e0
                self._v = 1e1 

            elif self.supervision == 'never':
                self._p = 1e-20
                self._pc = 1e-1  
                self._d = 1e-2   
                self._recon = 1e-2 
                self._vc = 0
                self._v = 1e1 

            loss_ =  self._p*photometric_loss_ + self._pc*pc_loss_ + self._recon*rgb_recover_loss_\
                        + self._d*depth_recover_loss_ +self._v*volume_loss_  + self._vc*vicon_supervision_loss_
            train_op = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.99).minimize(loss_)

        self.placeholders = {}
        self.placeholders['frame0_ph'] = frame0_ph
        self.placeholders['frame1_ph'] = frame1_ph
        self.placeholders['depth0_ph'] = depth0_ph
        self.placeholders['depth1_ph'] = depth1_ph
        self.placeholders['mask0_ph'] = mask0_ph
        self.placeholders['mask1_ph'] = mask1_ph
        self.placeholders['nan0_ph'] = nan0_ph
        self.placeholders['nan1_ph'] = nan1_ph
        self.placeholders['bbox0_ph'] = bbox0_ph
        self.placeholders['bbox1_ph'] = bbox1_ph
        self.placeholders['vicon0_ph'] = vicon0_ph
        self.placeholders['vicon1_ph'] = vicon1_ph
        self.placeholders['usingVicon0_ph'] = usingVicon0_ph
        self.placeholders['usingVicon1_ph'] = usingVicon1_ph
        self.placeholders['g_vr0_ph'] = g_vr0_ph
        self.placeholders['g_vr1_ph'] = g_vr1_ph

        self.tensors = {}
        self.tensors.update(self.placeholders)
        self.tensors['recovered_depth0'] = recovered_depth0_
        self.tensors['recovered_depth1'] = recovered_depth1_
        self.tensors['full_depth0'] = full_depth0_
        self.tensors['full_depth1'] = full_depth1_
        self.tensors['point_cloud0'] = point_cloud0_
        self.tensors['point_cloud1'] = point_cloud1_
        self.tensors['embed0'] = embed0_
        self.tensors['embed1'] = embed1_
        self.tensors['se3_0'] = se3_0_
        self.tensors['se3_1'] = se3_1_
        self.tensors['mask0'] = mask0_
        self.tensors['mask1'] = mask1_
        self.tensors['f_cam_motion'] = f_cam_motion_
        self.tensors['f_obj_motion'] = f_obj_motion_
        self.tensors['b_obj_motion'] = b_obj_motion_
        self.tensors['f_pix_pos'] = f_pix_pos_
        self.tensors['f_flow'] = f_flow_
        self.tensors['f_motion_map'] = f_motion_map_
        self.tensors['frame0_warped'] = frame0_warped_
        self.tensors['frame1_warped'] = frame1_warped_
        self.tensors['pc0_warped'] = pc0_warped_
        self.tensors['pc1_warped'] = pc1_warped_
        self.tensors['motioned_pc0'] = motioned_pc0_
        self.tensors['photometric_loss'] = photometric_loss_
        self.tensors['depth_recover_loss'] = depth_recover_loss_
        self.tensors['pc_loss'] = pc_loss_
        self.tensors['rgb_recover_loss'] = rgb_recover_loss_
        self.tensors['volume_loss'] = volume_loss_
        self.tensors['vicon_supervision_loss'] = vicon_supervision_loss_
        self.tensors['loss'] = loss_
        
        self.operators = {}
        self.operators['train_op'] = train_op
        
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        
        print('trainable variables are:')
        print(tf.trainable_variables())

    def test(self, saveFigure = False):
        sess = self.sess
        saver = self.saver
        placeholders = self.placeholders
        tensors = self.tensors
        task_name = self.task_name
        batch_size = self.batch_size
        scale = self.scale

        saver.restore(sess, self.weight_dir+'/u_net.ckpt')
        pose_stream = Pose_stream(task_name, batch_size, scale, mode = 'test', supervision = self.supervision)
        demo_list = pose_stream.demo_list
        batch_iter = pose_stream.iterator(batch_size = batch_size)
        writer = Writer(task_name, self.network_name+'_test')

        util.create_dir(self.output_dir, clear = True)  
        for demo_name in demo_list:
            util.create_dir(self.output_dir+'/'+demo_name+'/depth')
            util.create_dir(self.output_dir+'/'+demo_name+'/warped_img')
        
        self._error = 0
        global_step = 0
        one_cycle_loss = 0
        one_cycle_photo = 0
        one_cycle_pc = 0
        one_cycle_depth = 0
        one_cycle_rgb = 0
        one_cycle_vicon = 0
        one_cycle_volume = 0

        se3_dict = {}
        while True:
            try:
                frame0_batch, frame1_batch, \
                    depth0_batch, depth1_batch, \
                    mask0_batch, mask1_batch, \
                    nan0_batch, nan1_batch, \
                    vicon0_batch, vicon1_batch, \
                    usingVicon0_batch, usingVicon1_batch, \
                    g_vr0_batch, g_vr1_batch, \
                    bbox0_batch, bbox1_batch, \
                    demo_name = next(batch_iter)
                count = pose_stream._count
            except StopIteration:
                break
            feed_dict = {
                    placeholders['frame0_ph'] : frame0_batch,
                    placeholders['frame1_ph'] : frame1_batch,
                    placeholders['depth0_ph'] : depth0_batch,
                    placeholders['depth1_ph'] : depth1_batch,
                    placeholders['mask0_ph'] : mask0_batch,
                    placeholders['mask1_ph'] : mask1_batch,
                    placeholders['nan0_ph'] : nan0_batch,
                    placeholders['nan1_ph'] : nan1_batch,
                    placeholders['bbox0_ph'] : bbox0_batch,
                    placeholders['bbox1_ph'] : bbox1_batch,
                    placeholders['vicon0_ph'] : vicon0_batch,
                    placeholders['vicon1_ph'] : vicon1_batch,
                    placeholders['usingVicon0_ph'] : usingVicon0_batch,
                    placeholders['usingVicon1_ph'] : usingVicon1_batch,
                    placeholders['g_vr0_ph'] : g_vr0_batch,
                    placeholders['g_vr1_ph'] : g_vr1_batch
            }

            tensor_out = sess.run(tensors, feed_dict = feed_dict)
            one_cycle_loss += tensor_out['loss']
            one_cycle_photo += tensor_out['photometric_loss']
            one_cycle_pc += tensor_out['pc_loss']
            one_cycle_depth += tensor_out['depth_recover_loss']
            one_cycle_rgb += tensor_out['rgb_recover_loss']
            one_cycle_volume += tensor_out['volume_loss']
            one_cycle_vicon += tensor_out['vicon_supervision_loss']
            if saveFigure:
                self.subplot_depth(tensor_out, self.output_dir+'/'+demo_name+'/depth/%06d.png'%(count-1))
                self.subplot_warpedImg(tensor_out, self.output_dir+'/'+demo_name+'/warped_img/%06d.png'%(count-1))
                writer.write(str(self._error)+'\n')
            print('============')
            print(tensor_out['loss'])
            print(tensor_out['se3_0'])
            
            se3_0 = tensor_out['se3_0']
            se3_1 = tensor_out['se3_1']
            if demo_name not in se3_dict:
                se3_dict[demo_name] = []
                se3_dict[demo_name].append(se3_0)
            se3_dict[demo_name].append(se3_1)

        one_cycle_train = pose_stream._count
        log_string = '%d \t l: %.10f \t p: %.10f \t pc: %.10f \t vc: %.10f \t d: %.10f rgb: %.10f volume: %.10f \n'\
                            %(global_step, 
                                one_cycle_loss/one_cycle_train, one_cycle_photo/one_cycle_train,
                                one_cycle_pc/one_cycle_train, one_cycle_vicon/one_cycle_train,
                                one_cycle_depth/one_cycle_train, one_cycle_rgb/one_cycle_train, one_cycle_volume/one_cycle_train)
        writer.write(log_string)
        for demo_name, se3 in se3_dict.items():
            se3_dict[demo_name] = np.concatenate(se3, axis = 0)
        np.save(self.output_dir+'/se3_pose.npy', se3_dict)

    def train(self, continuous = False):
        sess = self.sess
        saver = self.saver
        placeholders = self.placeholders
        tensors = self.tensors
        operators = self.operators
        task_name = self.task_name
        batch_size = self.batch_size
        scale = self.scale

        pose_stream = Pose_stream(task_name, batch_size, scale, supervision = self.supervision)
        batch_iter = pose_stream.iterator(batch_size = batch_size)
        
        util.create_dir(self.weight_dir)
        if continuous:
            saver.restore(sess, self.weight_dir+'/u_net.ckpt')
            writer = Writer(task_name, self.network_name+'_train', append = True)
            util.create_dir(self.figure_dir+'/depth', clear = False)
            util.create_dir(self.figure_dir+'/warped_img', clear = False)
        else:
            sess.run(self.init)
            writer = Writer(task_name, self.network_name+'_train', append = False)
            util.create_dir(self.figure_dir+'/depth', clear = True)
            util.create_dir(self.figure_dir+'/warped_img', clear = True)
            
        global_step = 0
        one_cycle_loss = 0
        one_cycle_photo = 0
        one_cycle_pc = 0
        one_cycle_depth = 0
        one_cycle_rgb = 0
        one_cycle_vicon = 0
        one_cycle_volume = 0

        one_cycle_time = time.time()
        num_data = pose_stream.num_data
        print('number of data to train:%06d'%num_data)
        minimum_loss = 1e10
        while True:
            try:
                frame0_batch = []
                frame1_batch = []
                depth0_batch = []
                depth1_batch = []
                mask0_batch = []
                mask1_batch = []
                nan0_batch = []
                nan1_batch = []
                demo_batch = []
                bbox0_batch = []
                bbox1_batch = []
                vicon0_batch = []
                vicon1_batch = []
                usingVicon0_batch = []
                usingVicon1_batch = []
                g_vr0_batch = []
                g_vr1_batch = []
                try:
                    for i in range(self.batch_size):
                        frame0_sample, frame1_sample, \
                            depth0_sample, depth1_sample, \
                            mask0_sample, mask1_sample, \
                            nan0_sample, nan1_sample, \
                            vicon0_sample, vicon1_sample, \
                            usingVicon0_sample, usingVicon1_sample, \
                            g_vr0_sample, g_vr1_sample, \
                            bbox0_sample, bbox1_sample, demo_name = next(batch_iter)
                        frame0_batch.append(frame0_sample)
                        frame1_batch.append(frame1_sample)
                        depth0_batch.append(depth0_sample)
                        depth1_batch.append(depth1_sample)
                        mask0_batch.append(mask0_sample)
                        mask1_batch.append(mask1_sample)
                        nan0_batch.append(nan0_sample)
                        nan1_batch.append(nan1_sample)
                        vicon0_batch.append(vicon0_sample)
                        vicon1_batch.append(vicon1_sample)
                        usingVicon0_batch.append(usingVicon0_sample)
                        usingVicon1_batch.append(usingVicon1_sample)
                        g_vr0_batch.append(g_vr0_sample)
                        g_vr1_batch.append(g_vr1_sample)
                        bbox0_batch.append(bbox0_sample)
                        bbox1_batch.append(bbox1_sample)
                    frame0_batch = np.concatenate(frame0_batch, axis = 0)
                    frame1_batch = np.concatenate(frame1_batch, axis = 0)
                    depth0_batch = np.concatenate(depth0_batch, axis = 0)
                    depth1_batch = np.concatenate(depth1_batch, axis = 0)
                    mask0_batch = np.concatenate(mask0_batch, axis = 0)
                    mask1_batch = np.concatenate(mask1_batch, axis = 0)
                    nan0_batch = np.concatenate(nan0_batch, axis = 0)
                    nan1_batch = np.concatenate(nan1_batch, axis = 0)
                    vicon0_batch = np.concatenate(vicon0_batch, axis = 0)
                    vicon1_batch = np.concatenate(vicon1_batch, axis = 0)
                    usingVicon0_batch = np.concatenate(usingVicon0_batch, axis = 0)
                    usingVicon1_batch = np.concatenate(usingVicon1_batch, axis = 0)
                    g_vr0_batch = np.concatenate(g_vr0_batch, axis = 0 )
                    g_vr1_batch = np.concatenate(g_vr1_batch, axis = 0 )
                    bbox0_batch = np.concatenate(bbox0_batch, axis = 0)
                    bbox1_batch = np.concatenate(bbox1_batch, axis = 0)
                    
                except StopIteration:
                    ##
                    one_cycle_train = pose_stream._count
                    duration = time.time()-one_cycle_time
                    log_string = '%d \t %.2f \t l: %.10f \t p: %.10f \t pc: %.10f \t vc: %.10f \t d: %.10f rgb: %.10f window: %.10f \n'\
                            %(global_step, duration, 
                                one_cycle_loss/one_cycle_train, one_cycle_photo/one_cycle_train,
                                one_cycle_pc/one_cycle_train, one_cycle_vicon/one_cycle_train,
                                one_cycle_depth/one_cycle_train,
                                one_cycle_rgb/one_cycle_train, one_cycle_volume/one_cycle_train)
                    #log_string = str(global_step)+ '\t' + str(one_cycle_loss/one_cycle_train) + '\n'
                    writer.write(log_string)
                    
                    current_loss = one_cycle_loss/one_cycle_train
                    if minimum_loss > current_loss:
                        minimum_loss = current_loss
                        print(colored(log_string,'red'))
                    else:
                        print(log_string)

                    if global_step % (int(10000/num_data)+1) == 0 and global_step != 0:
                        saver.save(sess, self.weight_dir+'/u_net.ckpt')
                    if global_step % (int(1000/num_data)+1) == 0:
                        self.subplot_depth(tensor_out, self.figure_dir+'/depth/%06d.png'%global_step)
                        self.subplot_warpedImg(tensor_out, self.figure_dir+'/warped_img/%06d.png'%global_step)
                        
                    batch_iter = pose_stream.iterator(batch_size = 32)
                    one_cycle_loss = 0
                    one_cycle_photo = 0
                    one_cycle_pc = 0
                    one_cycle_depth = 0
                    one_cycle_rgb = 0
                    one_cycle_vicon = 0
                    one_cycle_volume = 0
                    one_cycle_time = time.time()
                    global_step +=1
                    continue

                feed_dict = {
                    placeholders['frame0_ph'] : frame0_batch,
                    placeholders['frame1_ph'] : frame1_batch,
                    placeholders['depth0_ph'] : depth0_batch,
                    placeholders['depth1_ph'] : depth1_batch,
                    placeholders['mask0_ph'] : mask0_batch,
                    placeholders['mask1_ph'] : mask1_batch,
                    placeholders['nan0_ph'] : nan0_batch,
                    placeholders['nan1_ph'] : nan1_batch,
                    placeholders['vicon0_ph'] : vicon0_batch,
                    placeholders['vicon1_ph'] : vicon1_batch,
                    placeholders['usingVicon0_ph'] : usingVicon0_batch,
                    placeholders['usingVicon1_ph'] : usingVicon1_batch,
                    placeholders['g_vr0_ph'] : g_vr0_batch,
                    placeholders['g_vr1_ph'] : g_vr1_batch,
                    placeholders['bbox0_ph'] : bbox0_batch,
                    placeholders['bbox1_ph'] : bbox1_batch
                }
                
                tensor_out, _ = sess.run([tensors, operators], feed_dict = feed_dict)
                one_cycle_loss += tensor_out['loss']
                one_cycle_photo += self._p*tensor_out['photometric_loss']
                one_cycle_pc += self._pc*tensor_out['pc_loss']
                one_cycle_depth += self._d*tensor_out['depth_recover_loss']
                one_cycle_rgb += self._recon*tensor_out['rgb_recover_loss']
                one_cycle_volume += self._v*tensor_out['volume_loss']
                one_cycle_vicon += self._vc*tensor_out['vicon_supervision_loss']
                
            except:
                traceback.print_exc()
                print('current task:'+self.task_name)
                print('saving weight...')
                #IPython.embed()
                time.sleep(3)
                saver.save(sess, self.weight_dir+'/u_net.ckpt')
                break

    def restore(self):
        sess = self.sess
        saver = self.saver
        #print_tensors_in_checkpoint_file(self.weight_dir+'/u_net.ckpt', all_tensors=True, tensor_name = '')
        var_list = checkpoint_utils.list_variables(self.weight_dir+'/u_net.ckpt')
        for var in var_list:
            print(var)
        saver.restore(sess, self.weight_dir+'/u_net.ckpt')
        
    def plot_output(self, tensor_out, file_path = ''):
        self.subplot_depth(tensor_out, file_path = file_path)
        self.subplot_warpedImg(tensor_out, file_path = file_path)

    def subplot_depth(self, tensor_out, file_path = ''):
        if not file_path:
            file_path = self.figure_dir+'/depth.png'

        depth0 = tensor_out['depth0_ph'][0,:,:]
        recovered_depth = tensor_out['recovered_depth0'][0,:]
        full_depth = tensor_out['full_depth0'][0,:]
        nan_mask = tensor_out['nan0_ph'][0,:]
        depth_diff = np.abs(full_depth - depth0)

        plt.figure(figsize = (10,10))
    
        plt.subplot(3,2,1)
        plt.imshow(self.normalize_depth(depth0), cmap ='gray')
        plt.axis('off')
        plt.title('sensor depth', fontsize=20)
        plt.subplot(3,2,2)
        plt.imshow(self.normalize_depth(recovered_depth), cmap ='gray')
        plt.axis('off')
        plt.title('recovered depth', fontsize=20)
        plt.subplot(3,2,3)
        plt.imshow(nan_mask, cmap ='gray')
        plt.axis('off')
        plt.title('nan mask', fontsize=20)
        plt.subplot(3,2,4)
        plt.imshow(self.normalize_depth(full_depth), cmap ='gray')
        plt.axis('off')
        plt.title('full depth', fontsize=20)

        plt.subplot(3,2,5)
        plt.pcolor(depth_diff, cmap ='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('depth_diff', fontsize=20)

        plt.savefig(file_path)
        plt.close()
    
    def subplot_warpedImg(self, tensor_out, file_path = ''):
        if not file_path:
            file_path = self.figure_dir+'/warpedImg.png'

        frame0 = tensor_out['frame0_ph'][0,:]
        frame1 = tensor_out['frame1_ph'][0,:]
        frame0_warped = tensor_out['frame0_warped'][0,:]
        frame1_warped = tensor_out['frame1_warped'][0,:]        
        mask0 =  tensor_out['mask0'][0,:,:,1:]
        mask1 = tensor_out['mask1'][0,:,:,1:]
        pc_loss = tensor_out['pc_loss']

        f_diff = np.sum(np.abs(frame1_warped-frame0), axis = 2)
        f_diff_ref = np.sum(np.abs(frame1-frame0), axis = 2)
        
        b_diff = np.sum(np.abs(frame0_warped-frame1), axis = 2)
        b_diff_ref = np.sum(np.abs(frame0-frame1), axis = 2)
        
        mask_ch = self.mask_ch
        f_diff_ex = np.expand_dims(f_diff, axis = 2)
        f_diff_tile = np.tile(f_diff_ex, [1,1,mask_ch])
        f_diff_masked = np.multiply(f_diff_tile, mask0)
        f_diff_masked = np.sum(f_diff_masked, axis = 2)

        f_diff_ref_ex = np.expand_dims(f_diff_ref, axis = 2)
        f_diff_ref_tile = np.tile(f_diff_ref_ex, [1,1,mask_ch])
        f_diff_ref_masked = np.multiply(f_diff_ref_tile, mask0)
        f_diff_ref_masked = np.sum(f_diff_ref_masked, axis = 2)

        b_diff_ex = np.expand_dims(b_diff, axis = 2)
        b_diff_tile = np.tile(b_diff_ex, [1,1,mask_ch])
        b_diff_masked = np.multiply(b_diff_tile, mask1)
        b_diff_masked = np.sum(b_diff_masked, axis = 2)

        b_diff_ref_ex = np.expand_dims(b_diff_ref, axis = 2)
        b_diff_ref_tile = np.tile(b_diff_ref_ex, [1,1,mask_ch])
        b_diff_ref_masked = np.multiply(b_diff_ref_tile, mask1)
        b_diff_ref_masked = np.sum(b_diff_ref_masked, axis = 2)

        f_diff_max = np.max([np.max(f_diff_masked), np.max(f_diff_ref_masked)])
        f_diff_min = np.min([np.min(f_diff_masked), np.min(f_diff_ref_masked)])

        b_diff_max = np.max([np.max(b_diff_masked), np.max(b_diff_ref_masked)])
        b_diff_min = np.min([np.min(b_diff_masked), np.min(b_diff_ref_masked)])

        plt.figure(figsize = (10,10))
        plt.subplot(5,2,1)
        plt.imshow(frame0)
        plt.title('frame0', fontsize=20)
        plt.axis('off')

        plt.subplot(5,2,2)
        plt.imshow(frame1)
        plt.title('frame1', fontsize=20)
        plt.axis('off')

        plt.subplot(5,2,3)
        plt.imshow(f_diff_masked, vmin = f_diff_min, vmax = f_diff_max)
        plt.title('f_difference', fontsize=20)
        plt.axis('off')

        plt.subplot(5,2,4)
        plt.imshow(f_diff_ref_masked, vmin = f_diff_min, vmax = f_diff_max)
        plt.title('f_difference_ref', fontsize=20)
        plt.axis('off')

        plt.subplot(5,2,5)
        plt.axis('off')

        plt.subplot(5,2,6)
        plt.axis('off')

        plt.subplot(5,2,7)
        plt.axis('off')

        plt.subplot(5,2,8)
        plt.axis('off')

        plt.subplot(5,2,9)
        plt.text(0.05, 0.7, 'bf warped(forward): %0.5f'%np.sum(f_diff_ref_masked), fontsize = 15)
        plt.text(0.05, 0.4, 'bf warped(backward): %0.5f'%np.sum(b_diff_ref_masked), fontsize = 15)
        plt.axis('off')

        plt.subplot(5,2,10)
        plt.text(0.05, 0.7, 'af warped(forward): %0.5f'%np.sum(f_diff_masked), fontsize = 15)
        plt.text(0.05, 0.4, 'af warped(backward): %0.5f'%np.sum(b_diff_masked), fontsize = 15)
        plt.text(0.05, 0.1, 'pc_loss'+str(pc_loss), fontsize = 15)
        plt.axis('off')

        plt.savefig(file_path)
        self._error += np.mean(f_diff_masked+b_diff_masked)
        plt.close()