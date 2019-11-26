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
from cv_bridge import CvBridge, CvBridgeError
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rosbag

from lib import util
from lib import vision
from lib.math import *

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('task_name')
        parser.add_argument('--clear', action = 'store_true')
    
        args = parser.parse_args()
        task_name = args.task_name
        CLEAR = args.clear
    except:
        traceback.print_exc()
        print(colored('type like: python3 ./read_bag.py [task_name] [option]'),'r')
        print(colored('ex: python3 ./read_bag.py task1 --clear'))
    
    task_dir = './data/'+task_name
    demo_dirs = sorted(os.listdir(task_dir))
    for demo_name in demo_dirs:
        ## 1. reindex rosbag file
        reindex_dir = task_dir + '/' + demo_name +'/reindex'
        bag_path = task_dir + '/' + demo_name +'/raw.bag.active' 
        util.create_dir(reindex_dir, clear = True)
        
        print(colored('reindex bag file','blue'))
        reindex_cmd = "rosbag reindex --output-dir=" + reindex_dir +" "+bag_path
        os.system(reindex_cmd)

        ## 2. extract infomation from reindexed bag
        print(colored('extract bag file','blue'))
        demo_dir = task_dir + '/' + demo_name
        rgb_dir = demo_dir + '/rgb_raw'
        depth_dir = demo_dir + '/depth_raw'
        vicon_dir = demo_dir + '/vicon_raw'
        object_names = util.load_txt('./configure/'+task_name+'_objects.txt')
        object_names += ['k_giant_ur3', 'k_zed']
        vicon_topics = ['/vicon/'+name+'/'+name for name in object_names ]
        
        util.create_dir(rgb_dir, clear = True)
        util.create_dir(depth_dir, clear = True)
        util.create_dir(vicon_dir, clear = True)

        for obj in object_names:
            util.create_dir(vicon_dir+'/'+obj, clear = True)

        ## reindex bag file    
        reindex_path = reindex_dir +'/raw.bag.active'
        bag = rosbag.Bag(reindex_path)

        ## log vicon
        bridge = CvBridge()
        ros_time0 = 0
        for topic, msg, t in bag.read_messages():
            ros_time = int( 1e-6*(1e9*msg.header.stamp.secs + msg.header.stamp.nsecs))
            if ros_time0 == 0:
                ros_time0 = ros_time
            ros_time = ros_time-ros_time0
            if topic == '/zed/zed_node/rgb/image_rect_color':
                rgb_file = rgb_dir + '/' + str(ros_time).zfill(10)
                cv_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(rgb_file+'.png', cv_rgb)
                plt_rgb = cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2GRAY)
                np.save(rgb_file+'.npy', plt_rgb)
                
            elif topic == '/zed/zed_node/depth/depth_registered':
                depth_file = depth_dir + '/' + str(ros_time).zfill(10)
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                cv2.imwrite(depth_file+'.png', cv_depth)
                np.save(depth_file+'.npy', cv_depth)
                
            elif topic in vicon_topics:
                obj_name = os.path.basename(topic)
                vicon_file = vicon_dir + '/' + obj_name+'/'+str(ros_time).zfill(10)
                
                posex = msg.transform.translation.x
                posey = msg.transform.translation.y
                posez = msg.transform.translation.z

                orix = msg.transform.rotation.x
                oriy = msg.transform.rotation.y
                oriz = msg.transform.rotation.z
                oriw = msg.transform.rotation.w
                quaternion = [posex, posey, posez, orix, oriy, oriz, oriw]
                se3 = quaternion_to_se3(quaternion)
                se3 = SE3_to_se3(se3_to_SE3(se3)) # rescale
                np.save(vicon_file+'.npy', se3)

        ## 3. extract rgb-images from long taken video and align depth and vicon data
        rgb_source_dir = demo_dir + '/rgb_raw'
        depth_source_dir = demo_dir + '/depth_raw'
     
        rgb_target_dir = demo_dir + '/rgb'
        depth_target_dir = demo_dir + '/depth'

        util.create_dir(rgb_target_dir, clear = True)
        util.create_dir(depth_target_dir, clear = True)
        print(colored('please check %s and inform us where to cut'%(demo_dir),'green') )
        begin = input(colored('where to begin? [enter number as type of nanoseconds]'))
        end = input(colored('where to end? [enter number as type of nanoseconds]'))
        begin = int(begin)
        end = int(end)
        #begin = 1574235759746797435
        #end = 1574235760578664782

        ## extract rgb
        rgb_time = []
        for rgb_path in sorted(glob.glob(rgb_source_dir+'/*.npy')):
            file_name = os.path.basename(rgb_path)
            file_name_no_extension = os.path.splitext(file_name)[0]
            time_index = int(file_name_no_extension)

            if (time_index >= begin) and (time_index <=end):
                source_file = rgb_source_dir+'/'+file_name_no_extension+'.npy'
                target_file = rgb_target_dir+'/'+str(len(rgb_time)).zfill(10)+'.npy'
                os.system('cp  %s %s'%(source_file, target_file))

                source_file = rgb_source_dir+'/'+file_name_no_extension+'.png'
                target_file = rgb_target_dir+'/'+str(len(rgb_time)).zfill(10)+'.png'
                os.system('cp  %s %s'%(source_file, target_file))
                rgb_time.append( int(time_index) )

            if time_index >end:
                break

        ## align depth
        depth_files = sorted(glob.glob(depth_source_dir+'/*.npy'))
        indicator = 0
        depth_time = []
        for d1, d2 in zip(depth_files[:-1], depth_files[1:]):
            d1_name = os.path.basename(d1)
            d2_name = os.path.basename(d2)
            d1_no_ext = os.path.splitext(d1_name)[0]
            d2_no_ext = os.path.splitext(d2_name)[0]
            
            d1_time = int(d1_no_ext)
            d2_time = int(d2_no_ext)

            if d1_time <= rgb_time[indicator] <= d2_time:
                if rgb_time[indicator] - d1_time <= d2_time- rgb_time[indicator]:
                    copy_file_no_ext = d1_no_ext
                else:
                    copy_file_no_ext =  d2_no_ext

                source_file = depth_source_dir+'/'+copy_file_no_ext+'.npy'
                target_file = depth_target_dir+'/'+str(indicator).zfill(10)+'.npy'
                os.system('cp  %s %s'%(source_file, target_file))

                source_file = depth_source_dir+'/'+copy_file_no_ext+'.png'
                target_file = depth_target_dir+'/'+str(indicator).zfill(10)+'.png'
                os.system('cp  %s %s'%(source_file, target_file))
                indicator += 1
                depth_time.append(int(copy_file_no_ext))
            else:
                pass

            if indicator >= len(rgb_time):
                break

        ## align vicon
        for obj in object_names:
            vicon_source_dir = demo_dir + '/vicon_raw/'+obj
            vicon_target_dir = demo_dir + '/vicon/'+obj
            util.create_dir(vicon_target_dir, clear = True)
        
            vicon_files = sorted(glob.glob(vicon_source_dir+'/*.npy'))
            indicator = 0
            vicon_time = []
            for v1, v2 in zip(vicon_files[:-1], vicon_files[1:]):
                v1_name = os.path.basename(v1)
                v2_name = os.path.basename(v2)
                v1_no_ext = os.path.splitext(v1_name)[0]
                v2_no_ext = os.path.splitext(v2_name)[0]
                
                v1_time = int(v1_no_ext)
                v2_time = int(v2_no_ext)

                if v1_time <= rgb_time[indicator] <= v2_time:
                    if rgb_time[indicator] - v1_time <= v2_time- rgb_time[indicator]:
                        copy_file_no_ext = v1_no_ext
                    else:
                        copy_file_no_ext =  v2_no_ext

                    source_file = vicon_source_dir+'/'+copy_file_no_ext+'.npy'
                    target_file = vicon_target_dir+'/'+str(indicator).zfill(10)+'.npy'
                    os.system('cp  %s %s'%(source_file, target_file))

                    indicator += 1
                    vicon_time.append(int(copy_file_no_ext))
                else:
                    pass

                if indicator >= len(rgb_time):
                    break

        ## save image animation
        config = util.load_yaml('./configure/'+task_name+'.yaml')
        fps = config['animation']['fps']
        img_dir = './data/'+task_name+'/'+demo_name+'/rgb'
        video_path = './data/'+task_name+'/'+demo_name+'/image_trajectory.avi'
        util.frame_to_video(img_dir, video_path,'png',fps=fps)()
           
        ## plot vicon trajectory
        se3_dict = {}
        for obj in object_names:
            vicon_dir = './data/'+task_name+'/'+demo_name+'/vicon/'+obj
            vicon_files = sorted(glob.glob(vicon_dir+'/*.npy'))
            se3_dict[obj] = []
            
            for f in vicon_files:
                se3 = np.load(f)
                se3_dict[obj].append(np.expand_dims(se3,0))    
            se3_dict[obj] = np.concatenate(se3_dict[obj],0)
            data_len = se3_dict[obj].shape[0]
    
        output_dir = './data/'+task_name+'/'+demo_name+'/vicon_traj'
        util.create_dir(output_dir, clear = True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b']

        se3_objects = {}
        se3_traj = {}
        obj_color = {}
        for i, obj in enumerate(object_names):
            se3_objects[obj] = vision.SE3object(angle_type = 'axis')
            se3_traj[obj] = np.zeros([0,3])
            obj_color[obj] = colors[i]

        for t in range(data_len):
            ax.clear()
            for obj in object_names:
                se3_object = se3_objects[obj]
                
                se3 = se3_dict[obj][t,:]
                SE3 = se3_to_SE3(se3)
                T = SE3[0:3,3]
                se3_traj[obj] = np.concatenate([se3_traj[obj], np.expand_dims(T,0)],0)
                
                ax.plot(se3_traj[obj][:,0],se3_traj[obj][:,1],se3_traj[obj][:,2], 
                        obj_color[obj], alpha = 0.5, linewidth = 4)
                se3_object.apply_pose(se3)
                se3_object.plot(ax, scale = 0.05, linewidth = 3) 

            util.set_axes_equal(ax)
            #ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_dir+'/%s.png'%str(t).zfill(10))
        video_path = output_dir+'/object_trajectory.avi'
        util.frame_to_video(output_dir, video_path, 'png', fps=fps)()




