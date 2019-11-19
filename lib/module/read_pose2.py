import cv2
import glob
import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython

from lib.math import *
from lib import vision
from lib import util

def projection(T, intrinsic):
    w = intrinsic.w
    h = intrinsic.h
    fx = intrinsic.fx
    fy = intrinsic.fy
    cx = intrinsic.cx
    cy = intrinsic.cy

    X = T[0] 
    Y = T[1]
    Z = T[2]+1e-10
    
    u = w*(fx*(X/Z)+cx)
    v = h*(fy*(Y/Z)+cy)

    return [u,v]

def read(config): 
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    supervision = config['pose']['supervision']
    scale = config['pose']['scale']
    fps = config['animation']['fps']
    pose_path = './output/pose/'+data_name
    
    data_dir = './data/'+data_name
    output_dir = './output/read_pose2/'+data_name
    se3_dict = np.load(pose_path+'/se3_pose.npy', allow_pickle = True).item()
    
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    intrinsic = vision.Zed_mini_intrinsic(scale = scale)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for demo_name, se3_traj in se3_dict.items():
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)
        img_list = sorted(glob.glob(data_dir+'/'+demo_name+'/image/*.npy'))
        
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))
        
        obj_idx = 0
        ref_frame = 0
        se30 = se3_traj[ref_frame,obj_idx,:]
        mask0 = np.load(mask_files[ref_frame])
        depth0 = np.load(depth_files[ref_frame]) 
        init_R = se3_to_SE3(se30)[0:3,0:3]
        com = vision.get_com(mask0, depth0, obj_idx, init_R = init_R)

        if supervision == 'never':
            g_c_com = se3_to_SE3(com)
            g_co = se3_to_SE3(se30)
            g_oc = inv_SE3(g_co)
            g_o_com = np.matmul(g_oc, g_c_com)
        else:
            # to do!!
            g_c_com = se3_to_SE3(com)
            g_co = se3_to_SE3(se30)
            g_oc = inv_SE3(g_co)
            g_o_com = np.matmul(g_oc, g_c_com)
        
        for t in range(len(se3_traj)):
            ax.clear()
            img = np.load(img_list[t])
            img = cv2.resize(img, None, fx = scale, fy = scale)
            ax.imshow(img/255.)  
            for obj_idx in range(nb_object):
                if supervision == 'never':
                    xi_co = se3_traj[t,obj_idx,:]
                    g_co = se3_to_SE3(xi_co)
                    g_c_com = np.matmul(g_co, g_o_com)
                    SE3 = np.copy(g_c_com)
                elif (supervision == 'both_ends') or (supervision == 'full'):
                    # to do!!!
                    g_c_com0 = np.copy(g_c_com)
                    SE3 = np.copy(g_c_com0)

                T = SE3[0:3,3]
                u,v = projection(T, intrinsic)
                ax.scatter(u,v, c = 'k')

                se3 = SE3_to_se3(SE3)
                obj.apply_pose(se3)
                s = 0.1 *10

                scaled_xbasis = 0.1*(obj.xbasis-obj.orientation)+obj.orientation
                scaled_ybasis = 0.1*(obj.ybasis-obj.orientation)+obj.orientation
                scaled_zbasis = 0.1*(obj.zbasis-obj.orientation)+obj.orientation
                    
                x_u, x_v = projection(scaled_xbasis, intrinsic)
                y_u, y_v = projection(scaled_ybasis, intrinsic)
                z_u, z_v = projection(scaled_zbasis, intrinsic)

                x_u_u = s*(x_u-u) + u
                x_v_v = s*(x_v-v) + v
                y_u_u = s*(y_u-u) + u
                y_v_v = s*(y_v-v) + v
                z_u_u = s*(z_u-u) + u
                z_v_v = s*(z_v-v) + v
                
                ax.plot([u, x_u_u], [v, x_v_v], c = 'r', linewidth = 3)
                ax.plot([u, y_u_u], [v, y_v_v], c = 'g', linewidth = 3)
                ax.plot([u, z_u_u], [v, z_v_v], c = 'b', linewidth = 3)
                ########################################
                ax.set_xlim([0, intrinsic.w])
                ax.set_ylim([intrinsic.h,0])

            fig.savefig(output_path+'/%06d.png'%t)
        video_path = output_path+'/se3_on_image.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=fps)()

