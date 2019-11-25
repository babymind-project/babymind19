import datetime
import glob
import os
import traceback

import IPython
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tflearn
from tflearn.layers.core import fully_connected, flatten
from tflearn.layers.conv import conv_2d, conv_2d_transpose
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization

from lib import network
from lib import util
from lib import vision
from lib import image
from lib.preprocess import *
from lib.learner import *

class Segnet_stream(Batch_stream):
    def __init__(self, config, mode = 'train'):
        self.mode = mode
        self.config = config
        self.task_name = task_name = config['task_name']  
        self.img_dict = {}
        self.depth_dict = {}
        self.mask_dict = {}

        if self.mode == 'train':
            self.data_dir = './output/'+task_name+'/preprocess'
            self.demo_list = os.listdir(self.data_dir)
            for demo in self.demo_list:
                demo_dir = self.data_dir+'/'+demo 
                self.img_dict[demo] = sorted(glob.glob(demo_dir+'/labeled_image/*.npy'))
                self.mask_dict[demo] = sorted(glob.glob(demo_dir+'/mask/*.npy'))
                self.depth_dict[demo] = []

                img_s_list = sorted(glob.glob('./output/'+task_name+'/label/'+demo+'/labeled_img/*.png'))
                for img_s in img_s_list:
                    filename = os.path.basename(img_s)
                    depth_s = './data/'+task_name+'/'+demo+'/depth/'+filename
                    depth_s = depth_s.replace('png','npy')
                    self.depth_dict[demo].append(depth_s)

        elif self.mode == 'test':
            self.data_dir = './data'+'/'+task_name
            self.demo_list = os.listdir(self.data_dir)
            for demo in self.demo_list:
                demo_dir = self.data_dir+'/'+demo 
                self.img_dict[demo] = sorted(glob.glob(demo_dir+'/rgb/*.npy'))
                self.depth_dict[demo] = sorted(glob.glob(demo_dir+'/depth/*.npy'))

        self.num_data = len(self.img_dict)  
        self.count = 0
        assert len(self.img_dict) > 0
        
    def iterator(self, batch_size):
        original_size = self.config['camera']['image_size']
        preprocess_size = self.config['preprocess']['image_size']
        scale = self.config['preprocess']['scale']
        assert original_size[0]*scale == preprocess_size[0]
        assert original_size[1]*scale == preprocess_size[1]

        self.count = 0
        #random_idx = np.random.randint(self.len)
        # loaded img : [0~255]
        # loaded mask : 
        if self.mode == 'train':
            for demo in self.demo_list:
                img_list = self.img_dict[demo]
                depth_list = self.depth_dict[demo]
                mask_list = self.mask_dict[demo]
                
                for img_path, depth_path, mask_path in zip(img_list, depth_list, mask_list):
                    img = np.load(img_path)
                    depth = np.load(depth_path)#[:,:,2]
                    mask = np.load(mask_path)

                    img = preprocess_img(img, scale = scale)
                    depth,_ = preprocess_depth(depth, scale = scale)
                    mask = preprocess_mask(mask, scale = scale)

                    img = np.expand_dims(img, axis = 0)
                    depth = np.expand_dims(depth, axis = 0)
                    mask = np.expand_dims(mask, axis = 0)
                    self.count +=1
                    yield img, depth, mask, demo

        elif self.mode =='test':
            for demo in self.demo_list:
                img_list = self.img_dict[demo]
                depth_list = self.depth_dict[demo]
                for img_path, depth_path in zip(img_list,depth_list):
                    img = np.load(img_path)
                    depth = np.load(depth_path)#[:,:,2]
                    
                    img = preprocess_img(img, scale = scale)
                    depth,_ = preprocess_depth(depth, scale = scale)
                    
                    img = np.expand_dims(img, axis = 0)
                    depth = np.expand_dims(depth, axis = 0)
                    mask = None
                    self.count +=1
                    yield img, depth, mask, demo


class Segment_network(Network):
    def __init__(self, config):
        self.tensors =[]
        self.operations =[]
        self.graph = []
        self.sess = []
        self.saver = []
        self.init = []
        
        self.config = config
        self.network_name = 'segment'
        self.task_name = config['task_name']
        self.depth_max = config['camera']['depth_max']
        self.depth_min = config['camera']['depth_min']
        self.img_size =  config['preprocess']['image_size']
        self.batch_size = config['segment']['batch_size']  
        self.learning_rate = config['segment']['learning_rate']
        self.mask_ch = config['object']['nb_object']
        self.loss_type = 'cross_entropy'
        self.fps = config['animation']['fps'] 

        self.weight_dir = './weight/'+self.task_name+'/'+self.network_name
        self.figure_dir = './figure/'+self.task_name+'/'+self.network_name
        self.output_dir = './output/'+self.task_name+'/'+self.network_name

    def normalize_depth(self, depth):
        MAX = self.depth_max/1000
        MIN = self.depth_min/1000
        # default MIN&MAX value is from zed-mini
        normalized_depth = (depth-MIN)/(MAX-MIN)
        #normalized_depth = np.clip(normalized_depth, 0, 1)
        return normalized_depth

    def build(self):
        img_size = self.img_size
        mask_ch = self.mask_ch
        loss_type = self.loss_type
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            with tf.variable_scope('segment'):
                img_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], 3))
                depth_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1]))
                mask_ph = tf.placeholder(dtype = tf.float32, shape=(None, img_size[0], img_size[1], mask_ch))
                
                depth_normalized_ =  self.normalize_depth(depth_ph)
                img_depth_concat = tf.concat([img_ph, tf.expand_dims(depth_normalized_, 3)],3)
                mask_logit, mask_pred, embed = network.u_net(img_depth_concat, mask_ch)
                
                if True:
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = mask_logit, labels = mask_ph))
                else:
                    # WTF!!! mse cannot train mask
                    loss = tf.reduce_mean(tf.square(mask_pred-mask_ph))
                train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        ## tensors and operations dictionary
        self.placeholders = {
            'img_ph' : img_ph,
            'depth_ph' : depth_ph,
            'mask_ph' : mask_ph
        }
        self.tensors = {
            'img_ph' : img_ph,
            'depth_ph' : depth_ph,
            'mask_ph': mask_ph,
            'mask_logit' : mask_logit,
            'mask_pred' : mask_pred,
            'loss' : loss
        }
        self.operations = {
            'train_op': train_op
        }
        ## sesion setting
        
    
    def train(self, continuous = False):
        sess = self.sess
        saver = self.saver
        placeholders = self.placeholders
        tensors = self.tensors
        operations = self.operations
        task_name = self.task_name
        batch_size = self.batch_size

        segnet_stream = Segnet_stream(self.config)
        batch_iter = segnet_stream.iterator(batch_size = batch_size)  
        util.create_dir(self.figure_dir, clear = True)
        util.create_dir(self.weight_dir, clear = False)

        if continuous:
            saver.restore(sess, self.weight_dir+'/u_net.ckpt')
            writer = Writer(task_name, self.network_name+'_train', append = True)
        else:
            sess.run(self.init)
            writer = Writer(task_name, self.network_name+'_train', append = False)
        global_step = 0
        one_cycle_loss = 0
        
        while True:
            try:
                try:
                    img_batch, depth_batch, mask_batch, demo_name = next(batch_iter)
                except StopIteration:
                    one_cycle_train = segnet_stream.count
                    log_string = str(global_step)+ '\t' + str(one_cycle_loss/one_cycle_train) +'\n'
                    writer.write(log_string)
                    
                    batch_iter = segnet_stream.iterator(batch_size = batch_size)
                    one_cycle_loss = 0
                    saver.save(sess, self.weight_dir+'/u_net.ckpt')
                    continue
                feed_dict = {placeholders['img_ph'] : img_batch,
                             placeholders['depth_ph'] : depth_batch,
                             placeholders['mask_ph'] : mask_batch}
                tensor_out, _  = sess.run([tensors, operations], feed_dict = feed_dict)
                global_step +=1
                one_cycle_loss += tensor_out['loss']
                print(tensor_out['loss'])
            except:
                traceback.print_exc()
                IPython.embed()
                break

    def test(self):
        sess = self.sess
        saver = self.saver
        placeholders = self.placeholders
        tensors = self.tensors
        operations = self.operations
        task_name = self.task_name
        batch_size = self.batch_size
        fps = self.fps

        saver.restore(sess, self.weight_dir+'/u_net.ckpt')
        test_tensors = {
            'img_ph' : tensors['img_ph'],
            'mask_pred': tensors['mask_pred']}

        segnet_stream = Segnet_stream(self.config, mode = 'test')
        batch_iter = segnet_stream.iterator(batch_size = batch_size)
        writer = Writer(task_name, self.network_name+'_test')
        util.create_dir(self.output_dir, clear = True) 

        while True:
            try:
                img_batch, depth_batch, _, demo_name = next(batch_iter)
                count = segnet_stream.count
            except StopIteration:
                break
            feed_dict = {placeholders['img_ph'] : img_batch,
                         placeholders['depth_ph'] : depth_batch}
            tensor_out  = sess.run(test_tensors, feed_dict = feed_dict)
            self.plot_output(tensor_out, demo_name, count)

        ## save image animation
        demo_list = os.listdir('./data/'+self.task_name)
        for demo_name in demo_list:
            img_path = self.output_dir+'/'+demo_name
            video_path = self.output_dir+'/'+demo_name+'/mask_video.avi'
            util.frame_to_video(img_path, video_path, 'png', fps = fps)()



    def restore(self):
        sess = self.sess
        saver = self.saver
        #print_tensors_in_checkpoint_file(self.weight_dir+'/u_net.ckpt', all_tensors=True, tensor_name = '')
        #var_list = checkpoint_utils.list_variables(self.weight_dir+'/u_net.ckpt')
        #for var in var_list:
        #    print(var)
        #saver.restore(sess, self.weight_dir+'/u_net.ckpt')
        saver.restore(sess, self.weight_dir+'/u_net.ckpt')
        
    def forward(self, img):
        sess = self.sess
        tensors = self.tensors
        placeholders = self.placeholders

        forward_tensors = {
            'img_ph' : tensors['img_ph'],
            'mask_pred' : tensors['mask_pred']
        }
        
        feed_dict = {
            placeholders['img_ph'] : np.expand_dims(img, axis = 0)
        }

        tensor_out = sess.run(forward_tensors, feed_dict = feed_dict)
        mask_pred =  tensor_out['mask_pred'][0,:,:,:]
        return mask_pred

    def plot_output(self, tensor_out, demo_name, count):
        img = tensor_out['img_ph'][0,:]
        mask = tensor_out['mask_pred'][0,:]
        util.create_dir(self.output_dir+'/'+demo_name)
        color = [0, 0, 1]
        
        mask_img = image.multichannel_to_image(mask)
        masked_img = image.apply_mask(img, mask_img, alpha = 0.5)
        
        plt.subplot(221)
        plt.imshow(img)

        plt.subplot(222)
        plt.imshow(mask_img)

        plt.subplot(223)
        plt.imshow(masked_img)

        plt.savefig(self.output_dir+'/'+demo_name+'/'+'%04d'%(count-1)+'.png')
        plt.close()

        np.save(self.output_dir+'/'+demo_name+'/'+'mask_%04d'%(count-1)+'.npy', mask)
    

    