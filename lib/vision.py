from __future__ import division

'''
Many modules come from 
https://github.com/waxz/sfm_net 
'''
import IPython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from lib.math import *

## camera
class Intrinsic():
    def __init__(self):
        self.h = 376
        self.w = 672
        # self.f = 
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.5
        self.cy = 0.5
        self.disto = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.v_fov = []
        self.h_fov = []
        self.d_fov = []
    
    # to do!!
    def adjust_size(self, h, w):
        # to do!!
        #prev_h = self.h
        #prev_w = self.w     
        #self.h = h
        #self.w = w
        pass
        
class Zed_intrinsic(Intrinsic):
    def __init__(self, scale = 1.):
        self.w = int(1280*scale)
        self.h = int(720*scale)
        self.f = 676.5778198
        self.fx = (self.f/self.w)*scale
        self.fy = (self.f/self.h)*scale
        
        self.cx = (647.502624511/self.w)*scale
        self.cy = (377.535461425/self.h)*scale

class Zed_mini_intrinsic(Intrinsic):
    def __init__(self, scale = 1.):
        self.w = int(672*scale)
        self.h = int(376*scale)
        self.f = 335.9676513671875 #332.6258
        self.fx = (self.f/self.w)*scale
        self.fy = (self.f/self.h)*scale
        
        self.cx = (322.11163330078125/self.w)*scale
        self.cy = (200.56185913085938/self.h)*scale
        

## vision algorithm
def np_cloud_transformer(depth, camera = 'zed_mini', scale = 1.):
    '''
    depth: [h,w]

    [(xi/w-cx)/fx,(yi/h-cy)/fy,1]
    next just 
    d*[(xi/w-cx)/fx,(yi/h-cy)/fy,1]
        to get [Xi,Yi,Zi]    
    '''
    if camera == 'zed':
        intrinsic = Zed_intrinsic(scale)
    elif camera == 'zed_mini':
        intrinsic = Zed_mini_intrinsic(scale)
    output_dim = 3
        
    cx = intrinsic.cx
    cy = intrinsic.cy
    fx = intrinsic.fx
    fy = intrinsic.fy
    
    height = depth.shape[0] #
    width  = depth.shape[1] #
    
    x_linspace = np.linspace(-cx,1-cx, width) # image height, array x
    y_linspace = np.linspace(-cy,1-cy, height) # imagw width, array y
    
    x_cord, y_cord = np.meshgrid(x_linspace, y_linspace)
    x_cord = np.reshape(x_cord,[-1])
    y_cord = np.reshape(y_cord,[-1])
    f_= np.expand_dims(np.ones_like(x_cord) ,0)
    
    x_= np.expand_dims(np.divide(x_cord, fx) ,0)
    y_= np.expand_dims(np.divide(y_cord, fy) ,0)
    grid = np.concatenate([ x_, y_, f_],0)

    depth = np.reshape(depth,[1,-1])
    depth = np.tile(depth, [output_dim,1])
    point_cloud = np.multiply(depth, grid)
    point_cloud = np.reshape(point_cloud, [output_dim, height, width])
    point_cloud = np.transpose(point_cloud, [1,2,0])
    return point_cloud



class Cloud_transformer():
    def __init__(self, intrinsic='zed_mini', scale=1., **kwargs):
        ## set intrinsic 
        if intrinsic == 'zed':
            self.intrinsic = Zed_intrinsic(scale)
        elif intrinsic == 'zed_mini':
            self.intrinsic = Zed_mini_intrinsic(scale)
        ## initialize
        self.output_dim = 3
        self.build()
        
    def build(self):
        self.cx_ = self.intrinsic.cx
        self.cy_ =self.intrinsic.cy
        self.fx_ =self.intrinsic.fx
        self.fy_ =self.intrinsic.fy
        self.cx = tf.constant(self.cx_, dtype=tf.float32)        
        self.cy = tf.constant(self.cy_, dtype=tf.float32)        
        self.fx = tf.constant(self.fx_, dtype=tf.float32)
        self.fy= tf.constant(self.fy_, dtype=tf.float32)

    def mesh_grid(self,width,height):
        # get 
        '''
        [(xi/w-cx)/fx,(yi/h-cy)/fy,1]
        next just 
        d*[(xi/w-cx)/fx,(yi/h-cy)/fy,1]
         to get [Xi,Yi,Zi]
        
        '''        
        # original code
        x_linspace=tf.linspace(-self.cx_,1-self.cx_, width) # image height, array x
        y_linspace=tf.linspace(-self.cy_,1-self.cy_, height) # imagw width, array y
        
        x_cord, y_cord = tf.meshgrid(x_linspace, y_linspace)
       
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)

        x_cord=tf.reshape(x_cord,[-1])
        y_cord=tf.reshape(y_cord,[-1])
        f_=tf.ones_like(x_cord)
        
        x_=tf.div(x_cord, self.fx)
        y_=tf.div(y_cord, self.fy)
        
        grid=tf.concat([x_, y_, f_],0)
        return grid
        
    def transform(self,x):
        #get input shape
        batch_size=tf.shape(x)[0]
        height=tf.shape(x)[1] #
        width=tf.shape(x)[2] #
        
        batch_size=tf.cast(batch_size,tf.int32)
        height = tf.cast(height,tf.int32)
        width = tf.cast(width,tf.int32)
        #grid
        grid = self.mesh_grid(width,height)
        
        grid=tf.expand_dims(grid,0)
        grid=tf.reshape(grid,[-1])
        
        grid_stack = tf.tile(grid, tf.stack([batch_size]))
        grid_stack=tf.reshape(grid_stack,[batch_size,3,-1])

        depth=tf.reshape(x,[batch_size,1,-1])
        depth=tf.concat([depth]*self.output_dim,1)
        
        point_cloud=tf.multiply(depth, grid_stack)
        return point_cloud

    def __call__(self, x):
        point_cloud=self.transform(x)
        return point_cloud

class Image_warper_backward():
    def __init__(self):
        pass
    
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def __call__(self, frame1, pos_2d_new):
        batch_size = tf.shape(frame1)[0]
        height = tf.shape(frame1)[1]
        width = tf.shape(frame1)[2]
        num_channels = tf.shape(frame1)[3]
        output_height = height
        output_width = width
        
        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        x_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
        x = tf.cast(tf.reshape(x_s, [-1]), dtype = 'float32')
        x = x*(width_float-1) 

        y_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1])
        y = tf.cast(tf.reshape(y_s, [-1]), dtype = 'float32')
        y = y*(height_float-1)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')

        zero = tf.zeros([], dtype='int32')
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        
        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1
        
        # add row and column
        flat_image = tf.reshape(frame1, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

        transformed_image = tf.add_n([area_a*pixel_values_a,
                            area_b*pixel_values_b,
                            area_c*pixel_values_c,
                            area_d*pixel_values_d])
        transformed_image = tf.reshape(transformed_image, shape = (-1,
                                                                    output_height,
                                                                    output_width,
                                                                    num_channels))
        return transformed_image

class Image_warper_forward():
    '''
    deprecated
    '''
    def __init__(self):
        self.transformed_image = []
        
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def __call__(self, frame0, pos_2d_new):
        batch_size = 1 #tf.shape(frame0)[0]
        height = 376 #tf.shape(frame0)[1]
        width = 672 #tf.shape(frame0)[2]
        num_channels = 3 #tf.shape(frame0)[3]
        output_height = height
        output_width = width
        
        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        x_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
        x = tf.cast(tf.reshape(x_s, [-1]), dtype = 'float32')
        x = x*(width_float)

        y_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1])
        y = tf.cast(tf.reshape(y_s, [-1]), dtype = 'float32')
        y = y*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')

        zero = tf.zeros([], dtype='int32')
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero+1, max_x+1)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero+1, max_y+1)
        
        x_ref = tf.range(1,width+1)
        y_ref = tf.range(1,height+1)
        x_ref, y_ref = tf.meshgrid(x_ref, y_ref)
        x_ref = self._repeat(x_ref, batch_size)
        y_ref = self._repeat(y_ref, batch_size)
        
        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1
        
        base_y_ref = base + y_ref*width
        indices_ref = base_y_ref + x_ref
        
        flat_image = tf.reshape(frame0, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_ref = tf.gather(flat_image, indices_ref)
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        
        #x1,y1 indicies_d : area_a*pixel_values_ref
        #x1,y0 indicies_c : area_b*pixel_values_ref
        #x0,y1 indicies_b : area_c*pixel_values_ref
        #x0,y0 indicies_a : area_d*pixel_values_ref
        
        pixel_x1y1 = area_a*pixel_values_ref
        pixel_x1y0 = area_b*pixel_values_ref
        pixel_x0y1 = area_c*pixel_values_ref
        pixel_x0y0 = area_d*pixel_values_ref

        if not self.transformed_image:
            self.transformed_image = tf.Variable(tf.zeros([batch_size*width*height, 3]), trainable = False)
        
        self.transformed_image = tf.assign(self.transformed_image, tf.zeros([batch_size*width*height, 3]))
        self.transformed_image = tf.scatter_update(self.transformed_image, indices_d, pixel_x1y1 )
        self.transformed_image = tf.scatter_add(self.transformed_image, indices_c, pixel_x1y0 )
        self.transformed_image = tf.scatter_add(self.transformed_image, indices_b, pixel_x0y1 )
        self.transformed_image = tf.scatter_add(self.transformed_image, indices_a, pixel_x0y0 )
        
        transformed_image = tf.reshape(self.transformed_image, shape = (-1,
                                                                    output_height,
                                                                    output_width,
                                                                    num_channels))
        return transformed_image


class Optical_transformer():
    def __init__(self, intrinsic='zed_mini', scale = 1., mask_ch = 1, input_type = 'sined_euler', **kwargs):
        ## set intrinsic 
        if intrinsic == 'zed':
            self.intrinsic = Zed_intrinsic(scale)
        elif intrinsic == 'zed_mini':
            self.intrinsic = Zed_mini_intrinsic(scale)
        ## initialize
        self.cx_ = self.intrinsic.cx
        self.cy_ = self.intrinsic.cy
        self.fx_ = self.intrinsic.fx
        self.fy_ = self.intrinsic.fy

        self.cx = tf.constant(self.cx_, dtype=tf.float32)        
        self.cy = tf.constant(self.cy_, dtype=tf.float32)        
        self.fx = tf.constant(self.fx_, dtype=tf.float32)
        self.fy = tf.constant(self.fy_, dtype=tf.float32)
        
        self.img_w=self.intrinsic.w
        self.img_h=self.intrinsic.h
        self.mask_size = mask_ch
        self.input_type = input_type
        so3_a=np.array([
            [0,-1,0,1,0,0,0,0,0],
            [1,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,1]
        ])

        so3_b=np.array([
            [0,0,1,0,0,0,-1,0,0],
            [1,0,0,0,0,0,0,0,1],
            [0,0,0,0,1,0,0,0,0]
        ])

        so3_y=np.array([
            [0,0,0,0,0,-1,0,1,0],
            [0,0,0,0,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,0]
        ])

        self.so3_a=self.np_tf(so3_a)
        self.so3_b=self.np_tf(so3_b)
        self.so3_y=self.np_tf(so3_y)
          
    def np_tf(self,array):
        return tf.constant(array,tf.float32)
    
    def build(self,cam_motion,obj_motion,x):                
        self.cam_motion=cam_motion
        self.obj_motion=obj_motion
        #self.mask_size=obj_motion[0].shape.as_list()[1]
        self.x_shape=x.shape.as_list()
            
    # tranformation     
    def so3_mat(self,sin):
        #input :sin a,sin b,sin y 
        #return : SO3
        sin=tf.expand_dims(sin,-1)
        cos=tf.sqrt(tf.ones_like(sin)-tf.square(sin))
        t=tf.concat([sin,cos,tf.ones_like(sin)],-1)
        t_a=tf.slice(t,[0,0,0],[-1,1,-1])
        t_b=tf.slice(t,[0,1,0],[-1,1,-1])
        t_y=tf.slice(t,[0,2,0],[-1,1,-1])
        t_a=tf.reshape(t_a,(-1,3))
        t_b=tf.reshape(t_b,(-1,3))
        t_y=tf.reshape(t_y,(-1,3))
        
        soa=tf.matmul(t_a,self.so3_a)
        soa=tf.reshape(soa,(-1,3,3))
        
        sob=tf.matmul(t_b,self.so3_b)
        sob=tf.reshape(sob,(-1,3,3))
        soy=tf.matmul(t_y,self.so3_y)
        soy=tf.reshape(soy,(-1,3,3))
   
        so3=tf.matmul(soa,  tf.matmul(sob,soy))
        return so3
        
    def rigid_motion(self,x, R, t):
        t=tf.expand_dims(t,-1)
        motion= tf.add(tf.matmul(R, x),t)        
        return motion
          
    def cam_motion_transform(self,x):
        t, sin =self.cam_motion
        R=self.so3_mat(sin)
        X=self.rigid_motion(x,R,t)
        return X

    def obj_motion_transform(self,x_input):
        t, rot, mask=self.obj_motion
        
        t=tf.reshape(t,(-1,3))
        x_in=tf.expand_dims(x_input,1)
        x_exp=tf.concat([x_in]*self.mask_size,1)
        x_=tf.reshape(x_exp,(-1,3,self.img_w*self.img_h))
        
        if self.input_type == 'sined_euler':
            sin=tf.reshape(rot,[-1,3])
            R=self.so3_mat(sin)
        elif self.input_type == 'SO3':
            R = tf.reshape(rot,[-1,3,3])

        x=self.rigid_motion(x_,R,t)
        x=tf.reshape(x,(-1,self.mask_size,3,self.img_w*self.img_h))
        x,motion_map=self.mask_motion(x,mask,x_exp)
        X=tf.add(x_input,x)

        return X, motion_map

    def mask_motion(self,x,mask,x_exp):
        mask = tf.transpose(mask, [0, 3, 1, 2])
        mask=tf.reshape(mask,(-1,self.mask_size,1,self.img_w*self.img_h))

        x=tf.subtract(x,x_exp)

        motion_map=tf.multiply(x,mask)
        x=tf.reduce_sum(motion_map,1)
        return x,motion_map
                
    def transform_2d(self,x):
        epsilon = 1e-5 # prevent nan
        x_3d=tf.slice(x,(0,0,0),(-1,1,self.img_w*self.img_h))
        y_3d=tf.slice(x,(0,1,0),(-1,1,self.img_w*self.img_h))
        z_3d=tf.slice(x,(0,2,0),(-1,1,self.img_w*self.img_h))
        x_z=tf.div(x_3d,z_3d+epsilon)
        y_z=tf.div(y_3d,z_3d+epsilon)
        
        # x_2d=tf.multiply(self.img_w,tf.add(tf.multiply(self.cf,x_z),self.cx))
        # y_2d=tf.multiply(self.img_h,tf.add(tf.multiply(self.cf,y_z),self.cy))
        
        x_2d=tf.add(tf.multiply(self.fx, x_z), self.cx)
        y_2d=tf.add(tf.multiply(self.fy, y_z), self.cy)
        
        #pos_2d_new=tf.concat([x_2d, y_2d],1)
        pos_2d_new=tf.concat([y_2d, x_2d],1)
        return pos_2d_new

    def get_flow(self,pos_2d_new):
        x_linspace = tf.linspace(0.,1.,int(self.img_w))
        y_linspace = tf.linspace(0.,1.,int(self.img_h))
        
        #y_linspace,x_linspace = tf.meshgrid( y_linspace,x_linspace)
        x_linspace, y_linspace = tf.meshgrid( x_linspace, y_linspace)
        
        x_linspace = tf.reshape(x_linspace, [1,-1])
        y_linspace = tf.reshape(y_linspace, [1,-1])
        pos_ori=tf.concat([y_linspace,x_linspace],0)
        flow=tf.subtract(pos_2d_new,pos_ori)
        return flow

    def __call__(self,x,cam_motion,obj_motion,):
        self.build( cam_motion,obj_motion,x)
        point_cloud, motion_map=self.obj_motion_transform(x)
        # make enable when you use cam_motion
        #point_cloud=self.cam_motion_transform(point_cloud)
        pix_pos=self.transform_2d(point_cloud)
        flow=self.get_flow(pix_pos)
        motion_map=tf.reshape(motion_map,(-1, self.mask_size, 3, self.img_h*self.img_w))
        return pix_pos, flow, point_cloud, motion_map

class Smoother():
    def __init__(self, img_size, kernel=[[1,2,1],[0,0,0],[-1,-2,-1]],order=1):
        self.kernel=np.array(kernel)
        self.order=order
        self.img_h = img_size[0]
        self.img_w = img_size[1]
    def build(self, field_c):
        v_kernel=self.kernel
        h_kernel=self.kernel.T
        h_init=tf.keras.initializers.Constant(value=h_kernel)
        v_init=tf.keras.initializers.Constant(value=v_kernel)
        
        self.conv_h=tf.keras.layers.Conv2D(filters=field_c,kernel_size=3,strides=1,kernel_initializer=h_init,padding='same')
        self.conv_h.trainable=False
        self.conv_v=tf.keras.layers.Conv2D(filters=field_c,kernel_size=3,strides=1,kernel_initializer=v_init,padding='same')
        self.conv_v.trainable=False

    def compute_gradient(self,field):
        loss_v=self.conv_v(field)
        loss_h=self.conv_h(field)
        gradient_loss=loss_h+loss_v
        return gradient_loss

    def compute_loss(self,field):
        f1_gradient_loss=self.compute_gradient(field)
        if self.order==1:
            loss=tf.reduce_mean(tf.abs(f1_gradient_loss),-1)
            loss=tf.reduce_mean(loss)
        if self.order==2:
            f2_gradient_loss=self.compute_gradient(f1_gradient_loss)
            loss=tf.reduce_mean(tf.abs(f2_gradient_loss),-1)
            loss=tf.reduce_mean(loss)
        return loss

    def get_loss(self,field,loss_type=None,reuse=False):
        with tf.variable_scope(loss_type,reuse=reuse):
            if loss_type=='flow':
                field=tf.keras.layers.Permute((2,1))(field)
                field=tf.reshape(field,(-1, self.img_h, self.img_w,2))
                field_c=field.shape.as_list()[1]

            field_c=field.shape.as_list()[-1]
            self.build(field_c)
            loss=self.compute_loss(field)
            return loss

class SE3object():
    def __init__(self, init_state = [0,0,0,0,0,0], angle_type = 'euler'):
        '''
        init_state: numpy array shape with 12 
        init_state[0:3] : xyz  
        init_state[3:6] : roll, pitch, yaw
        '''
        self.angle_type = angle_type
        self.orientation = np.asarray([0,0,0])
        self.xbasis = np.asarray([1,0,0])
        self.ybasis = np.asarray([0,1,0])
        self.zbasis = np.asarray([0,0,1])
        self.init_state = init_state
        

        if self.angle_type == 'euler':
            R = euler_to_SO3(init_state[3:6])
            T = init_state[0:3]
        elif self.angle_type == 'axis':
            SE3 = se3_to_SE3(init_state)
            R = SE3[0:3,0:3]
            T = SE3[0:3,3]
        self.apply_RT(R,T)

    def apply_RT(self, R, T):
        ori_ = to_homo(self.orientation)
        xbasis_ = to_homo(self.xbasis)
        ybasis_ = to_homo(self.ybasis)
        zbasis_ = to_homo(self.zbasis)

        SE3 = RT_to_SE3(R,T)
        new_ori_ = np.matmul(SE3, np.expand_dims(ori_, axis = 1))
        new_xbasis_ = np.matmul(SE3, np.expand_dims(xbasis_, axis = 1))
        new_ybasis_ = np.matmul(SE3, np.expand_dims(ybasis_, axis = 1))
        new_zbasis_ = np.matmul(SE3, np.expand_dims(zbasis_, axis = 1))
        
        new_ori = un_homo(new_ori_)
        new_xbasis = un_homo(new_xbasis_)
        new_ybasis = un_homo(new_ybasis_)
        new_zbasis = un_homo(new_zbasis_)

        self.orientation = np.squeeze(new_ori, axis = 1)
        self.xbasis = np.squeeze(new_xbasis, axis = 1)
        self.ybasis = np.squeeze(new_ybasis, axis = 1)
        self.zbasis = np.squeeze(new_zbasis, axis = 1)

    def apply_SE3(self, SE3):
        ori_ = to_homo(self.orientation)
        xbasis_ = to_homo(self.xbasis)
        ybasis_ = to_homo(self.ybasis)
        zbasis = to_homo(self.zbasis)
        
        new_ori_ = np.matmul(SE3, np.expand_dims(ori_, axis = 1))
        new_xbasis_ = np.matmul(SE3, np.expand_dims(xbasis_, axis = 1))
        new_ybasis_ = np.matmul(SE3, np.expand_dims(ybasis_, axis = 1))
        new_zbasis_ = np.matmul(SE3, np.expand_dims(zbasis_, axis = 1))
        
        new_ori = un_homo(new_ori_)
        new_xbasis = un_homo(new_xbasis_)
        new_ybasis = un_homo(new_ybasis_)
        new_zbasis = un_homo(new_zbasis_)

        self.orientation = np.squeeze(new_ori, axis = 1)
        self.xbasis = np.squeeze(new_xbasis, axis = 1)
        self.ybasis = np.squeeze(new_ybasis, axis = 1)
        self.zbasis = np.squeeze(new_zbasis, axis = 1)

    def apply_pose(self, new_pose):
        self.orientation = np.asarray([0,0,0])
        self.xbasis = np.asarray([1,0,0])
        self.ybasis = np.asarray([0,1,0])
        self.zbasis = np.asarray([0,0,1])
        
        SE3 = se3_to_SE3(new_pose)
        R = SE3[0:3,0:3]
        T = SE3[0:3,3] 
        assert self.angle_type =='axis'
        '''
        if self.angle_type == 'euler':
            R = euler_to_SO3(new_pose[3:6])
        elif self.angle_type == 'axis':
            R = omega_to_SO3(new_pose[3:6])
        '''
        self.apply_RT(R,T)

    def plot(self, ax, scale = 1, linewidth = 5):
        '''
        ax should 3d axes 
        '''
        ori = self.orientation
        scaled_xbasis = scale*(self.xbasis-ori)+ori
        scaled_ybasis = scale*(self.ybasis-ori)+ori
        scaled_zbasis = scale*(self.zbasis-ori)+ori

        xbar = np.concatenate([np.expand_dims(ori,1), np.expand_dims(scaled_xbasis,1)], axis = 1)
        ybar = np.concatenate([np.expand_dims(ori,1), np.expand_dims(scaled_ybasis,1)], axis = 1)
        zbar = np.concatenate([np.expand_dims(ori,1), np.expand_dims(scaled_zbasis,1)], axis = 1)

        ax.plot(xbar[0,:], xbar[1,:], xbar[2,:], color='r', linewidth = linewidth)
        ax.plot(ybar[0,:], ybar[1,:], ybar[2,:], color='g', linewidth = linewidth)
        ax.plot(zbar[0,:], zbar[1,:], zbar[2,:], color='b', linewidth = linewidth)


def get_com(mask, depth, obj_idx, init_R = np.eye(3)):
    h = mask.shape[0]
    w = mask.shape[1]

    mask = mask[:,:, obj_idx]
    depth = depth[:,:,0:3]
    
    mask_rowed = np.sum(mask,1)
    mask_columned = np.sum(mask, 0)

    row_range = np.arange(h)
    column_range = np.arange(w)

    row_cg = np.sum(np.multiply(row_range, mask_rowed))/np.sum(mask_rowed)
    column_cg = np.sum(np.multiply(column_range, mask_columned))/np.sum(mask_columned)

    row_cg = int(row_cg)
    column_cg = int(column_cg)
    #print('('+str(row_cg)+','+str(column_cg)+')' )
    
    #fo = intrinsic.f
    #ho = 376 * scale
    #wo = 672 * scale
    
    T = depth[row_cg,column_cg,:]
    T[0] = T[0]#*(fo/wo)
    T[1] = T[1]#*(fo/ho)
    T[2] = T[2]
    
    R = init_R
    SE3 = RT_to_SE3(R, T)
    init_xi = SE3_to_se3(SE3)
    return init_xi


## vision algorithm
class Projection():
    def __init__(self, intrinsic='zed', scale=1., **kwargs):
        ## set intrinsic 
        if intrinsic == 'zed':
            self.intrinsic = Zed_intrinsic()
        elif intrinsic == 'zed_mini':
            self.intrinsic = Zed_mini_intrinsic(scale)
        ## initialize
        self.output_dim = 3
        self.build()
        
    def build(self):
        self.w_ = self.intrinsic.w
        self.h_ = self.intrinsic.h
        self.cx_ = self.intrinsic.cx
        self.cy_ =self.intrinsic.cy
        self.fx_ =self.intrinsic.fx
        self.fy_ =self.intrinsic.fy

        self.w = tf.constant(self.w_, dtype=tf.float32)
        self.h = tf.constant(self.h_, dtype=tf.float32)                
        self.cx = tf.constant(self.cx_, dtype=tf.float32)        
        self.cy = tf.constant(self.cy_, dtype=tf.float32)        
        self.fx = tf.constant(self.fx_, dtype=tf.float32)
        self.fy = tf.constant(self.fy_, dtype=tf.float32)

    def get_uv(self,se3):
        # get 
        '''
        [X,Y,Z] = d*[(xi/w-cx)/fx,(yi/h-cy)/fy,1]
        [x,y] = [w*(fx*X/Z+cx), h*(fy*Y/Z+cy)]
        '''
        X = se3[:,:,0] #[N,K]
        Y = se3[:,:,1]
        Z = se3[:,:,2] 

        x = self.w*(self.fx*tf.div(X,Z)+self.cx) #[N,K]
        y = self.h*(self.fy*tf.div(Y,Z)+self.cy) #[N,K]
        return tf.concat([tf.expand_dims(x,2), tf.expand_dims(y,2)],2)
         

