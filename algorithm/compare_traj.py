import glob
import os

import IPython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import io
from scipy.signal import medfilt
from scipy.optimize import minimize
from termcolor import colored

from lib.math import *
from lib import util
from lib import vision

def load_vicon(vicon_dir):
    vicon_files = sorted(glob.glob(vicon_dir+'/*.npy'))
        
    traj = []
    for f in vicon_files:
        se3 = np.load(f)
        traj.append(np.expand_dims(se3,0))
    return np.concatenate(traj,0)


def compare(config, LOAD = False):
    '''
    vicon data is required!
    execute 'read_bag' first
    '''

    task_name = config['task_name']
    fps = config['animation']['fps']
    object_list = util.load_txt('./configure/%s_objects.txt'%task_name)
    nb_object = len(object_list)

    data_dir = './data/'+task_name
    pose_dir = './output/'+task_name+'/read_pose'
    output_dir = './output/'+task_name+'/compare_traj'
    util.create_dir(output_dir, clear = False)
    
    ## load data
    demo_list = os.listdir('./data/'+task_name)
    for demo in demo_list:
        rgb_demo_dir = data_dir+'/'+demo+'/rgb'
        cam_demo_dir = data_dir+'/'+demo+'/vicon/k_zed'       
        output_demo_dir = output_dir +'/'+demo
        util.create_dir(output_demo_dir+'/traj', clear = True)
        util.create_dir(output_demo_dir+'/plot', clear = True)

        # g_vc1 : vicon to camera(vicon_pose) = cam pose by vicon
        # g_c1c2 : camera(vicon_pose) to camera center(center of image plane)
        # g_c2o1 : camera center to object(vision coordinates) = object pose by vision
        # g_o1o2 : object(vision coordinates) to object(vicon coordinates) 
        # g_vo2_gt : vicon to object(vicon coordinates) = object pose by vicon
        assert len(object_list) == 1 ## to do : multiple object
        obj_demo_dir = data_dir+'/'+demo+'/vicon/'+object_list[0] ## to do : multiple object 
        se3_c2o1 = np.load(pose_dir+'/'+demo+'/pose_traj.npy')[:,0,:] ## to do : multiple object
        se3_vc1 = load_vicon(cam_demo_dir)
        se3_vo2_gt = load_vicon(obj_demo_dir)
        
        # make vicon orientation = [0,0,0]
        demo_len = len(se3_c2o1)
        g_vo2_gt_0 = se3_to_SE3(se3_vo2_gt[0,:])
        R0 = g_vo2_gt_0[0:3,0:3]
        T0 = np.zeros(3)
        g0_inv = RT_to_SE3(np.transpose(R0),T0)
        for t in range(demo_len):
            g_vo2_gt_t = se3_to_SE3(se3_vo2_gt[t,:])
            new_g_vo2_gt_t = np.matmul(g_vo2_gt_t, g0_inv)
            new_se3_vo2_gt_t = SE3_to_se3(new_g_vo2_gt_t)
            se3_vo2_gt[t,:] = new_se3_vo2_gt_t

        x0 = np.random.rand(12)
        optimize_len = int(len(se3_c2o1))
        def objective_fn(x0):
            # g_vc1 : vicon to camera(vicon_pose) = se3_cam
            # g_c1c2 : camera(vicon_pose) to camera center(center of image plane)
            # g_c2o1 : camera center to object(vision coordinates) = se3_vision
            # g_o1o2 : object(vision coordinates) to object(vicon coordinates) 
            g_c1c2 = se3_to_SE3(x0[0:6])
            g_o1o2 = se3_to_SE3(x0[6:12])
            loss = 0
            for t in range(optimize_len):
                g_c2o1_t = se3_to_SE3(se3_c2o1[t,:])
                g_vc1_t = se3_to_SE3(se3_vc1[t,:])
                g_vo2 = np.matmul(g_vc1_t, np.matmul(g_c1c2, np.matmul(g_c2o1_t, g_o1o2)))
                g_vo2_gt = se3_to_SE3(se3_vo2_gt[t,:])
                se3_vo2 = SE3_to_se3(g_vo2)
                loss += np.sum(np.square(se3_vo2[:]-se3_vo2_gt[t,:])) #----------------
                #loss += np.sum(np.square(g_vo2[0:3,3]-g_vo2_gt[0:3,3]))
                #loss += np.sum(np.square(g_vo2-g_vo2_gt)) ----------------------
            return loss
        print(colored('initial_loss:'+str(objective_fn(x0)),'blue'))
        
        if LOAD:
            result = np.load(output_demo_dir+'/optimization.npy', allow_pickle = True).item()
            cmd = input('Optimization more? [y/n]')
            if cmd =='y':
                x0 = result.x
                result = minimize(objective_fn, 
                    x0, 
                    method='BFGS', 
                    options={'disp': True})
            else:
                pass
        else:
            #'''
            result = minimize(objective_fn, 
                    x0, 
                    method='BFGS', 
                    options={'disp': True})
            #'''
            '''
            result = minimize(objective_fn, 
                    x0, 
                    method='BFGS', 
                    tol=1e-8,
                    options={'gtol': 1e-8, 'disp': True})
            '''
            
            np.save(output_demo_dir+'/optimization.npy',result)
        print(colored('optimized_loss:'+str(objective_fn(result.x)),'blue'))           
        g_c1c2 = se3_to_SE3(result.x[0:6])
        g_o1o2 = se3_to_SE3(result.x[6:12])

        obj_vicon = vision.SE3object(np.zeros(6), angle_type = 'axis')
        obj_vision = vision.SE3object(np.zeros(6), angle_type = 'axis')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        loss = 0
        position_error = []
        rotation_error = []
        vicon_traj = []
        vision_traj = []
        vicon_euler = []
        vision_euler = []

        T_vo2 = np.zeros((0,3))
        T_vo2_gt = np.zeros((0,3))
        
        g_c2o1_0 = se3_to_SE3(se3_c2o1[0,:])
        g_vc1_0 = se3_to_SE3(se3_vc1[0,:])
        g_vo2_0 = np.matmul(g_vc1_0, np.matmul(g_c1c2, np.matmul(g_c2o1_0, g_o1o2)))
        g_vo2_0_gt = se3_to_SE3(se3_vo2_gt[0,:])

        R_target = g_vo2_0_gt[0:3,0:3]
        R_ori = g_vo2_0[0:3,0:3] 
        T_ori = g_vo2_0[0:3,3]
        SO3_align = np.matmul(R_target, np.transpose(R_ori)) 
        SE3_align = np.matmul(inv_SE3(g_vo2_0), g_vo2_0_gt)

        for t in range(demo_len):
            g_c2o1_t = se3_to_SE3(se3_c2o1[t,:])
            g_vc1_t = se3_to_SE3(se3_vc1[t,:])
            g_vo2_t = np.matmul(g_vc1_t, np.matmul(g_c1c2, np.matmul(g_c2o1_t, g_o1o2)))
            #g_vo2_t[0:3,0:3] = np.matmul(SO3_align, g_vo2_t[0:3,0:3])
            #g_vo2_t = np.matmul(g_vo2_t, SE3_align)

            se3_vo2_t_gt = SE3_to_se3(se3_to_SE3(se3_vo2_gt[t,:]))
            g_vo2_t_gt = se3_to_SE3(se3_vo2_t_gt)
            se3_vo2_t = SE3_to_se3(g_vo2_t)
            
            T_vo2 = np.concatenate( [T_vo2, np.expand_dims(g_vo2_t[0:3,3],0)], 0)
            T_vo2_gt = np.concatenate( [T_vo2_gt, np.expand_dims(g_vo2_t_gt[0:3,3],0)], 0)

            ax.clear()
            obj_vicon.apply_pose(se3_vo2_t)
            obj_vision.apply_pose(se3_vo2_t_gt)
            obj_vicon.plot(ax, scale = 0.015, linewidth = 3)
            obj_vision.plot(ax, scale = 0.015, linewidth = 3)
            ax.plot(T_vo2_gt[:t,0], T_vo2_gt[:t,1], T_vo2_gt[:t,2], '--', color = 'r', alpha = 0.5, linewidth = 4)
            ax.plot(T_vo2[:,0], T_vo2[:,1], T_vo2[:,2], color = 'g', alpha = 0.5, linewidth = 3)
                
            util.set_axes_equal(ax)
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            fig.savefig(output_demo_dir+'/traj/%05d.png'%t)

            ## rescale (||w|| = 1)
            #se3_vo2_t[3:6] =  (1./np.sqrt(np.sum(np.square(se3_vo2_t[3:6]))))*se3_vo2_t[3:6]
            #se3_vo2_t_gt[3:6] = (1./np.sqrt(np.sum(np.square(se3_vo2_t_gt[3:6]))))*se3_vo2_t_gt[3:6]

            loss += np.sqrt(np.sum(np.square(se3_vo2_t_gt-se3_vo2_t)))
            position_error.append( np.expand_dims( np.sqrt(np.sum(np.square(g_vo2_t_gt[0:3,3]-g_vo2_t[0:3,3]))),0))
            rotation_error.append( np.expand_dims( np.sqrt(np.sum(np.square(se3_vo2_t_gt[3:6]-se3_vo2_t[3:6]))),0))
            
            vicon_traj.append(np.expand_dims(se3_vo2_t_gt,0))
            vision_traj.append(np.expand_dims(se3_vo2_t,0))
            vicon_euler.append(np.expand_dims( R_to_euler(g_vo2_t_gt[0:3,0:3]),0))
            vision_euler.append(np.expand_dims( R_to_euler(g_vo2_t[0:3,0:3]),0))

            if t == 0:
                total_translation = 0
                total_rotation = 0
                prev_g_vo = g_vo2_t_gt
                prev_se3_vo = se3_vo2_t_gt
            else:
                total_translation += np.sqrt(np.sum(np.square(g_vo2_t_gt[0:3,3]-prev_g_vo[0:3,3])))
                total_rotation += np.sqrt(np.sum(np.square(se3_vo2_t_gt[3:6]-prev_se3_vo[3:6])))
                prev_g_vo = g_vo2_t_gt
                prev_se3_vo = se3_vo2_t_gt
        plt.close()

        ## save loss
        loss = loss/demo_len
        position_error = np.sum(position_error)/demo_len #np.concatenate(position_error,0)
        rotation_error = np.sum(rotation_error)/demo_len #np.concatenate(rotation_error,0)
        
        vicon_traj = np.concatenate(vicon_traj,0)
        vision_traj = np.concatenate(vision_traj,0)
        vicon_euler = np.concatenate(vicon_euler,0)
        vision_euler = np.concatenate(vision_euler,0)
        
        np.savetxt(output_demo_dir+'/loss.txt',[loss])
        np.savetxt(output_demo_dir+'/position_error.txt',[position_error])
        np.savetxt(output_demo_dir+'/rotation_error.txt',[rotation_error])
        np.savetxt(output_demo_dir+'/total_translation.txt',[total_translation])
        np.savetxt(output_demo_dir+'/total_rotation.txt',[total_rotation])
        np.savetxt(output_demo_dir+'/vicon_traj.txt',vicon_traj)
        np.savetxt(output_demo_dir+'/vision_traj.txt',vision_traj)

        ## save plot
        fig = plt.figure()
        ymins = []
        ymaxs = []
        axes = []
        scales = []
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.plot(np.arange(demo_len), vicon_traj[:,i], '--', color = 'r', alpha = 0.5, linewidth = 4)
            ax.plot(np.arange(demo_len), vision_traj[:,i], color = 'g', alpha = 0.5, linewidth = 3)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
            axes.append(ax)
            scales.append(ymax-ymin)
        ymin = min(ymins)
        ymax = max(ymaxs)
        scale_max = max(scales)
        for ax, ymin, ymax in zip(axes, ymins, ymaxs):
            center = (ymin+ymax)/2
            ax.set_ylim([center-scale_max/2, center+scale_max/2])   
        fig.savefig(output_demo_dir+'/v_component.png')
        plt.close()

        fig = plt.figure()
        ymins = []
        ymaxs = []
        axes = []
        scales = []
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.plot(np.arange(demo_len), vicon_traj[:,i+3], '--', color = 'r', alpha = 0.5, linewidth = 4)
            ax.plot(np.arange(demo_len), vision_traj[:,i+3], color = 'g', alpha = 0.5, linewidth = 3)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
            axes.append(ax)
            scales.append(ymax-ymin)
        ymin = min(ymins)
        ymax = max(ymaxs)
        scale_max = max(scales)
        for ax, ymin, ymax in zip(axes, ymins, ymaxs):
            center = (ymin+ymax)/2
            ax.set_ylim([center-scale_max/2, center+scale_max/2])   
        fig.savefig(output_demo_dir+'/w_component.png')
        plt.close()

        ## translation and rotation
        fig = plt.figure()
        ymins = []
        ymaxs = []
        axes = []
        scales = []
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.plot(np.arange(demo_len), T_vo2_gt[:,i], '--', color = 'r', alpha = 0.5, linewidth = 4)
            ax.plot(np.arange(demo_len), T_vo2[:,i], color = 'g', alpha = 0.5, linewidth = 3)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
            axes.append(ax)
            scales.append(ymax-ymin)
        ymin = min(ymins)
        ymax = max(ymaxs)
        scale_max = max(scales)
        for ax, ymin, ymax in zip(axes, ymins, ymaxs):
            center = (ymin+ymax)/2
            ax.set_ylim([center-scale_max/2, center+scale_max/2])   
        fig.savefig(output_demo_dir+'/translation_component.png')
        plt.close()

        fig = plt.figure()
        ymins = []
        ymaxs = []
        axes = []
        scales = []
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.plot(np.arange(demo_len), vicon_euler[:,i], '--', color = 'r', alpha = 0.5, linewidth = 4)
            ax.plot(np.arange(demo_len), vision_euler[:,i], color = 'g', alpha = 0.5, linewidth = 3)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
            axes.append(ax)
            scales.append(ymax-ymin)
        ymin =  min(ymins)
        ymax =  max(ymaxs)
        scale_max = max(scales)
        for ax, ymin, ymax in zip(axes, ymins, ymaxs):
            center = (ymin+ymax)/2
            ax.set_ylim([center-scale_max/2, center+scale_max/2])   
        fig.savefig(output_demo_dir+'/rotation_component.png')
        plt.close()
