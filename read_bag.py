from __future__ import division

'''
This is python2  code due to the dependecy of rospy to python2
'''

import glob
import os
import sys
from termcolor import colored
import rosbag

from scipy import io
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from lib.math import *
from lib import util

if __name__ == '__main__':
    try:
        task_name = sys.argv[1] 
    except:
        print(colored('Type like: python ./read_bag.py [task_name]','red'))

    data_dir = './data/'+task_name
    output_dir = './output/vicon/'+task_name
    demo_list = os.listdir(data_dir)
    
    for demo in demo_list:
        bag_list = glob.glob(data_dir+'/'+demo+'/*.bag')
        print(demo)
        print(bag_list)
        output_path = output_dir+'/'+demo
        util.create_dir(output_path, clear = True) 

        ### read vicon and save at './output/vicon/task_name/demo_name/objet_name_pose.npy'
        all_pose = []
        for bag_file in bag_list:
            bag = rosbag.Bag(bag_file)
            object_name = bag_file.split('/')[-1].split('.')[0]

            original_pose = []
            se3 = []
            pose = []
            
            prev_t = 0
            for topic, msg, t in bag.read_messages():
                ros_time = msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs
                if prev_t == 0:
                    prev_t = ros_time
                log_time = ros_time-prev_t
                
                posex = msg.pose.position.x
                posey = msg.pose.position.y
                posez = msg.pose.position.z

                orix = msg.pose.orientation.x
                oriy = msg.pose.orientation.y
                oriz = msg.pose.orientation.z
                oriw = msg.pose.orientation.w
                quaternion = [posex, posey, posez, orix, oriy, oriz, oriw]
                se3_t = quaternion_to_se3(quaternion)
                #se3.append(np.expand_dims(se3_t,0))
                se3_t = SE3_to_se3(se3_to_SE3(se3_t)) # rescale
                se3.append(np.expand_dims([ros_time, se3_t[0], se3_t[1],se3_t[2],se3_t[3],se3_t[4],se3_t[5]],0))

                pose_t = se3_to_SE3(se3_t)[0:3,3]
                #pose.append(np.expand_dims([log_time, posex, posey, posez, w[0], w[1], w[2]],0))
                pose.append(np.expand_dims([ros_time, pose_t[0], pose_t[1],pose_t[2]],0))

                original_pose_t = [posex, posey, posez]
                original_pose.append(np.expand_dims(original_pose_t,0))
            se3 = np.concatenate(se3, 0)
            pose = np.concatenate(pose , 0)

            original_pose = np.concatenate(original_pose,0)
            np.save(output_path+'/'+object_name+'_pose.npy', se3)
            np.savetxt(output_path+'/'+object_name+'_pose.txt', se3)
            all_pose.append(pose)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        nb_object = len(all_pose)
        for pose in all_pose:
            ax.plot(pose[:,1], pose[:,2], pose[:,3])
        util.set_axes_equal(ax)
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')
        
        plt.savefig(output_path+'/pose_path.png')
        plt.close()
        
        ### align vicon with image save at './output/vicon/task_name/demo_name/clipped_objetname_pose.npy'
        data_demo_dir = data_dir+'/'+demo 
        vicon_demo_dir = output_dir + '/' + demo
        cam_pos0 = np.load(data_demo_dir+'/camera_position0.npy', allow_pickle = True)
        cam_pos1 = np.load(data_demo_dir+'/camera_position1.npy', allow_pickle = True)
        cut_point = np.loadtxt(data_demo_dir+'/cut.txt')

        # open vicon
        vicon_traj_list = []
        obj_files = sorted(glob.glob(vicon_demo_dir+'/*.npy'))
        assert len(obj_files) == 1
        for obj_file in obj_files:
            vicon_traj_loaded = np.load(obj_file, allow_pickle = True)
            vicon_traj_list.append(vicon_traj_loaded)
        # open cam time
        cam_start_time = cam_pos0[0] 
        cam_finish_time = cam_pos1[0]
        # cut point
        total_img = cut_point[0]
        beginning_cut = cut_point[1]
        endding_cut = cut_point[2]
        # modify cam start time
        play_time = cam_finish_time-cam_start_time
        cam_start_time = cam_start_time+play_time*(beginning_cut/(total_img-1))
        cam_finish_time = cam_finish_time - play_time*(endding_cut/(total_img-1))
        ## open len vision
        len_vision = int(total_img - beginning_cut - endding_cut)
        try:
            assert len(glob.glob(data_demo_dir+'/image/*.png')) == len_vision
        except:
            print('error of'+data_demo_dir)
            print('number of image:'+str(len(glob.glob(data_demo_dir+'/image/*.png'))))
            print('len_vision'+str(len_vision))
            raise NotImplementedError
        v_t = 1
        for i in range(nb_object):
            vicon_traj = vicon_traj_list[i]
            vicon_time = vicon_traj[:,0]

            vicon_start_idx = 0
            vicon_finish_idx = len(vicon_time)
            for i in range(len(vicon_time)):
                if vicon_time[i] < cam_start_time:
                    vicon_start_idx = i
                if vicon_time[i] < cam_finish_time:
                    vicon_finish_idx = i
            vicon_start_idx += 1

            clipped_vicon_time = vicon_time[vicon_start_idx:vicon_finish_idx]
            clipped_vicon = vicon_traj[vicon_start_idx:vicon_finish_idx, 1:7]
            vision_time = np.arange(clipped_vicon_time[0], clipped_vicon_time[-1], (clipped_vicon_time[-1]-clipped_vicon_time[0])/len_vision)

            vicon_se3 = np.zeros((0,6))
            for t in range(len_vision):
                for tt in range(v_t-1, len(clipped_vicon_time)):
                    if clipped_vicon_time[tt] > vision_time[t]:
                        break
                v_t = tt
                vicon_se3  = np.concatenate([vicon_se3,  np.expand_dims(clipped_vicon[v_t,:],0) ], 0)

        # need modifying for multiple object
        vicon_se3_all_object = np.expand_dims(vicon_se3,1)
        np.save(output_path+'/aligned.npy', vicon_se3_all_object)
        
        save_data = {'se3': vicon_se3_all_object[:,0,:], 'time': vision_time}
        io.savemat(output_path+'/trajectories.mat', save_data)
