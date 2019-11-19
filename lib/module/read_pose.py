import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython

from lib.math import *
from lib import vision
from lib import util

def read(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    fps = config['animation']['fps']

    se3_path = './output/pose/'+data_name+'/se3_pose.npy'
    demo_list = os.listdir('./data/'+data_name)
    output_dir = './output/read_pose/'+data_name
    
    se3_dict = np.load(se3_path, allow_pickle = True).item()
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color = ['g','r']
    # draw trajectory w.r.t. camera
    for demo_name, se3_traj in se3_dict.items():
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)

        depth = np.load(depth_files[0])
        mask = np.load(mask_files[0])  

        se30 = se3_traj[0,0,:]
        SE30 = se3_to_SE3(se30)
        T0 = SE30[0:3,3]
        T_traj = np.expand_dims(T0,0)
        # modify for multiple object
        for t in range(len(se3_traj)):
            ax.clear()
            for obj_idx in range(nb_object):
                se3 = se3_traj[t,obj_idx,:]
                SE3 = se3_to_SE3(se3)
                T = SE3[0:3,3]
                T_traj = np.concatenate([T_traj, np.expand_dims(T,0)], 0)
                ax.plot(T_traj[:,0], T_traj[:,1], T_traj[:,2],
                        color[obj_idx], alpha = 0.5, linewidth = 4)
                obj.apply_pose(se3)
                obj.plot(ax, scale = 0.05, linewidth = 3) 

            util.set_axes_equal(ax)
            #ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_path+'/%06d.png'%t)

        np.save(output_path+'/pose_traj.npy', se3_traj)
        video_path = output_path+'/object_trajectory.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=fps)()

