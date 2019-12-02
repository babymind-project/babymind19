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

from lib.math import *
from lib import util
from lib import vision

def load_vicon(vicon_dir):
    vicon_files = sorted(glob.glob('*.npy'))
    
    traj = []
    for f in vicon_files:
        se3 = np.load(f)
        traj.append(np.expand_dims(se3,0))
    return np.concatenate(traj,0)


def compare(config):
    '''
    vicon data is required!
    execute 'read_bag' first
    '''

    task_name = config['take_name']
    fps = config['animation']['fps']
    object_list = util.load_txt('./configure/%s_objects.txt'%task_name)
    nb_object = len(object_list)

    data_dir = './data/'+task_name
    pose_dir = './output/'+task_name+'/read_pose'
    output_dir = './output/'+task_name+'/compare_traj'
    util.create_dir(output_dir, clear = True)
    
    ## load data
    demo_list = os.listdir('./data/'+task_name)
    for demo in demo_list:
        rgb_demo_dir = data_dir+'/'+demo+'/rgb'
        cam_demo_dir = data_dir+'/'+demo+'/vicon/k_zed'        
        assert len(object_list) == 1 ## to do : multiple object
        obj_demo_dir = data_dir+'/'+demo+'/vicon/'+object_list[0] 
        se3_vision = np.load(pose_dir+'/'+demo+'/pose_traj.npy')
        se3_vicon = load_vicon(obj_demo_dir)
        se3_cam = laod_vicon(cam_demo_dir)

        IPython.embed()
        sys.exit()

        obj_vicon = vision.SE3object(np.zeros(6), angle_type = 'axis')
        obj_vision = vision.SE3object(np.zeros(6), angle_type = 'axis')

        
        
            for t in range(len(vision_traj)): # ah
                for tt in range(v_t-1, len(clipped_vicon_time)):
                    if clipped_vicon_time[tt] > vision_time[t]:
                        break
                v_t = tt
                #print('vision_time:%04.04f'%vision_time[t])
                #print('vicon_time:%04.04f'%clipped_vicon_time[v_t])
                #print('\n')
                
                ## time
                vicon_T  = se3_to_SE3(clipped_vicon[v_t,:])
                vision_T = se3_to_SE3(vision_traj[t,:])
                vicon_plot  = np.concatenate([vicon_plot,  np.expand_dims(vicon_T[0:3,3],0) ], 0)
                vision_plot = np.concatenate([vision_plot, np.expand_dims(vision_T[0:3,3],0)], 0)

                vicon_se3_element = SE3_to_se3(se3_to_SE3(clipped_vicon[v_t,:]))
                vicon_se3  = np.concatenate([vicon_se3,  np.expand_dims(vicon_se3_element,0) ], 0)
                vision_se3 = np.concatenate([vision_se3, np.expand_dims( SE3_to_se3(vision_T),0)], 0)
                
            plt.close('all')
            ## align translation
            x0 = np.random.rand(12)
            optimize_len = int(len(vision_plot))
            ########################## for task3 (occlusion)
            #if task_name == 'task3':
            #    optimize_len = 100
            def objective_fn(x0):
                SE30 = se3_to_SE3(x0[0:6])
                SE31 = se3_to_SE3(x0[6:12])
                loss = 0
                for t in range(optimize_len):
                    transformed = np.matmul(np.matmul(SE30, se3_to_SE3(vision_se3[t,:])),SE31)
                    transformed_se3 = SE3_to_se3(transformed)
                    loss += np.sum(np.square(transformed_se3[:]-vicon_se3[t,:]))
                    
                    #transformed = un_homo( np.matmul(np.matmul(SE30, to_homo(vision_plot[t,:])),SE31))
                    #loss += np.sum(np.square(transformed-vicon_plot[t,:]))
                return loss
            print(demo)
            print('initial_loss:'+str(objective_fn(x0)))
            if False:
                result = np.load(output_dir+'/'+demo+'/optimization.npy').item()        
            else:
                #'''
                result = minimize(objective_fn, 
                                    x0, 
                                    method='BFGS', 
                                    tol=1e-7,
                                    options={'gtol': 1e-6, 'disp': True})
                
                #'''
                '''
                result = minimize(objective_fn, 
                                    x0, 
                                    method='Nelder-Mead',
                                    options={'maxiter':1})
                '''
                np.save(output_dir+'/'+demo+'/optimization.npy',result)
            print('optimized_loss:'+str(objective_fn(result.x)))   
            print(result.x)
            
            SE30 = se3_to_SE3(result.x[0:6])   
            SE31 = se3_to_SE3(result.x[6:12])   
            ### align orientation
            #'''
            
            vision_SE30 = np.matmul(np.matmul(SE30, se3_to_SE3(vision_se3[0,:])),SE31)
            vicon_SE30 = se3_to_SE3(vicon_se3[0,:])
            R_target = vicon_SE30[0:3,0:3]
            R_ori = vision_SE30[0:3,0:3] 
            T_ori = vision_SE30[0:3,3]
            
            SO3_align = np.matmul(R_target, np.transpose(R_ori)) 
            
            #####################################################
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            transformed_vision_se3 = np.zeros((0,6))
            transformed_vision_SE3 = np.zeros((0,4,4))
            transformed_vision_plot = np.zeros((0,3))
            
            loss = 0
            position_error = []
            rotation_error = []
            total_position = []
            total_rotation = []
            for t in range(len(vicon_se3)):
                ## time
                vicon_se3_t = vicon_se3[t,:]
                vision_se3_t = vision_se3[t,:]
                vicon_T_t = se3_to_SE3(vicon_se3[t,:])
                vision_T_t = se3_to_SE3(vision_se3[t,:])
                ############################### not alignment!!!!!!!!!
                
                #if supervision == 'never': 
                #    se3_rc = np.asarray([0.333653152, -0.19545771, -0.41434446, -0.16426212, 0.4854613, -0.28981152],dtype = np.float32)
                #    g_rc = se3_to_SE3(se3_rc)
                #    _, g_vr = util.load_vicon(data_demo_dir+'/camera_position0.npy')
                #    SE3 = np.matmul(g_vr, g_rc)
                    
                vision_T_t = np.matmul(np.matmul(SE30, vision_T_t),SE31)
                #vision_T_t[0:3,0:3] = np.matmul(SO3_align, vision_T_t[0:3,0:3])
                vision_se3_t = SE3_to_se3(vision_T_t)
                #IPython.embed()

                transformed_vision_se3 = np.concatenate([transformed_vision_se3, np.expand_dims(SE3_to_se3(vision_T_t),0)], 0)
                transformed_vision_plot = np.concatenate([transformed_vision_plot, np.expand_dims(vision_T_t[0:3,3],0)], 0)
                transformed_vision_SE3 = np.concatenate([transformed_vision_SE3, np.expand_dims(vision_T_t,0)],0)

                loss += np.sqrt(np.sum(np.square(SE3_to_se3(vicon_T_t)-SE3_to_se3(vision_T_t))))
                position_error.append( np.expand_dims( np.sqrt(np.sum(np.square(vicon_T_t[0:3,3]-vision_T_t[0:3,3]))),0))
                rotation_error.append( np.expand_dims( np.sqrt(np.sum(np.square(vicon_se3_t[3:6]-vision_se3_t[3:6]))),0))
                total_position.append(np.expand_dims(vicon_T_t[0:3,3],0))
                total_rotation.append(np.expand_dims(vicon_se3_t[3:6],0))

                ax.clear()
                obj_vicon.apply_pose(vicon_se3_t)
                obj_vision.apply_pose(vision_se3_t)
                obj_vicon.plot(ax, scale = 0.015, linewidth = 3)
                obj_vision.plot(ax, scale = 0.015, linewidth = 3)
                ##
                ax.plot(vicon_plot[:t,0], vicon_plot[:t,1], vicon_plot[:t,2], '--', color = 'r', alpha = 0.5, linewidth = 4)
                ax.plot(transformed_vision_plot[:,0], transformed_vision_plot[:,1], transformed_vision_plot[:,2], color = 'g', alpha = 0.5, linewidth = 3)
                ###
                util.set_axes_equal(ax)
                ax.set_xlabel('x(m)')
                ax.set_ylabel('y(m)')
                ax.set_zlabel('z(m)')
                fig.savefig(output_dir+'/'+demo+'/%05d.png'%t)
            plt.close() 

            loss = loss/len(vision_traj)
            position_error = np.sum(position_error)/len(vision_traj) #np.concatenate(position_error,0)
            rotation_error = np.sum(rotation_error)/len(vision_traj) #np.concatenate(rotation_error,0)
            
            total_position = np.concatenate(total_position,0)
            total_rotation = np.concatenate(total_rotation,0)
            
            np.savetxt(output_dir+'/'+demo+'/loss.txt',[loss])
            np.savetxt(output_dir+'/'+demo+'/position_error.txt',[position_error])
            np.savetxt(output_dir+'/'+demo+'/rotation_error.txt',[rotation_error])
            np.savetxt(output_dir+'/'+demo+'/total_position.txt',total_position)
            np.savetxt(output_dir+'/'+demo+'/total_rotation.txt',total_rotation)

            #################### save data for jigang
            pickle_dict = {'time': np.transpose(vision_time), 'se3': transformed_vision_se3, 'SE3':  transformed_vision_SE3}
            io.savemat(output_dir+'/'+demo+'/pickle/trajectories.mat', pickle_dict)

            #####################################################
            if len(vision_time) > len(transformed_vision_se3):
                vision_time = vision_time[:len(transformed_vision_se3)]

            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax1.plot(clipped_vicon_time, clipped_vicon[:,0],'--')
            ax1.plot(vision_time, transformed_vision_se3[:,0])
            ax1.set_title('se3[0:3]')

            ax2 = fig.add_subplot(312)
            ax2.plot(clipped_vicon_time, clipped_vicon[:,1],'--')
            ax2.plot(vision_time, transformed_vision_se3[:,1])

            ax3 = fig.add_subplot(313)
            ax3.plot(clipped_vicon_time, clipped_vicon[:,2],'--')
            ax3.plot(vision_time, transformed_vision_se3[:,2])
            fig.savefig(output_dir+'/'+demo+'/plot/xyz.png')
            #####################################################
            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax1.plot(clipped_vicon_time, clipped_vicon[:,3],'--')
            ax1.plot(vision_time, transformed_vision_se3[:,3])
            ax1.set_title('se3[3:6]')

            ax2 = fig.add_subplot(312)
            ax2.plot(clipped_vicon_time, clipped_vicon[:,4],'--')
            ax2.plot(vision_time, transformed_vision_se3[:,4])

            ax3 = fig.add_subplot(313)
            ax3.plot(clipped_vicon_time, clipped_vicon[:,5],'--')
            ax3.plot(vision_time, transformed_vision_se3[:,5])
            fig.savefig(output_dir+'/'+demo+'/plot/wxwywz.png')
            plt.close()
            #####################################################
            diff_vicon = (clipped_vicon[1:,:]-clipped_vicon[:-1,:])/np.expand_dims(clipped_vicon_time[1:]-clipped_vicon_time[:-1], 1)
            diff_vision = (transformed_vision_se3[1:,:] - transformed_vision_se3[:-1,:])/np.expand_dims(vision_time[1:]-vision_time[:-1],1)

            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax1.plot(clipped_vicon_time[:-1], diff_vicon[:,0],'--')
            ax1.plot(vision_time[:-1], diff_vision[:,0])
            ax1.set_title('diff_se3[0:3]')
            ax1.set_ylim([-0.5,0.5])

            ax2 = fig.add_subplot(312)
            ax2.plot(clipped_vicon_time[:-1], diff_vicon[:,1],'--')
            ax2.plot(vision_time[:-1], diff_vision[:,1])
            ax2.set_ylim([-0.5,0.5])
            
            ax3 = fig.add_subplot(313)
            ax3.plot(clipped_vicon_time[:-1], diff_vicon[:,2],'--')
            ax3.plot(vision_time[:-1], diff_vision[:,2])
            ax3.set_ylim([-0.5,0.5])
            
            fig.savefig(output_dir+'/'+demo+'/plot/diff_xyz.png')
            plt.close()
            #####################################################
            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax1.plot(clipped_vicon_time[:-1], diff_vicon[:,3],'--')
            ax1.plot(vision_time[:-1], diff_vision[:,3])
            ax1.set_title('diff_se3[0:3]')
            ax1.set_ylim([-0.5,0.5])

            ax2 = fig.add_subplot(312)
            ax2.plot(clipped_vicon_time[:-1], diff_vicon[:,4],'--')
            ax2.plot(vision_time[:-1], diff_vision[:,4])
            ax2.set_ylim([-0.5,0.5])

            ax3 = fig.add_subplot(313)
            ax3.plot(clipped_vicon_time[:-1], diff_vicon[:,5],'--')
            ax3.plot(vision_time[:-1], diff_vision[:,5])
            ax3.set_ylim([-0.5,0.5])
            
            fig.savefig(output_dir+'/'+demo+'/plot/diff_wxwywz.png')
            plt.close()
            #####################################################

        video_path = output_dir+'/'+demo+'/se3_compare.avi'
        util.frame_to_video(output_dir+'/'+demo, video_path, 'png', fps=fps)()


            
            

'''
def apply_transform(pose, transform):
    # pose : [N,K,6]
    # transform : ([3,], [4,])
    T = transform[0]
    quaternion = transform[1]
    R = quaternion_to_R(quaternion)
    G = RT_to_SE3(R,T)

    vicon_to_cam_R = np.asarray([[0, 0.5, 0.8660254],
                                 [1, 0, 0],
                                 [0, 0.8660254, -0.5]])
    vicon_to_cam_T = np.asarray([0,0,0.5])
    vicon_to_cam_G = RT_to_SE3(vicon_to_cam_R, vicon_to_cam_T)
    G = np.matmul(G,vicon_to_cam_G)

    traj = []
    for t in range(pose.shape[0]):
        traj_k = []
        for k in range(pose.shape[1]):
            obj_SE3 = se3_to_SE3(pose[t,k,:])
            transformed_SE3 = np.matmul(G,obj_SE3)
            
            transformed_se3 = SE3_to_se3(transformed_SE3)
            traj_k.append(np.expand_dims(transformed_se3,0))
        traj_k = np.concatenate(traj_k, 0)
        traj.append(np.expand_dims(traj_k,0))
    traj = np.concatenate(traj, 0)
    return traj

def read_traj(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    demo_list = os.listdir('./data/'+data_name)

    pose_dir = './output/read_pose/'+data_name
    transform_dir = './data/'+data_name
    output_dir = './output/read_traj/'+data_name
    util.create_dir(output_dir, clear = True)

    fig  = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    color = ['g', 'r']
    robot = vision.SE3object(np.zeros(6), angle_type = 'axis')
    cam = vision.SE3object(np.zeros(6), angle_type = 'axis')
    vicon_to_cam_R = np.asarray([[0, 0.5, 0.8660254],
                                 [1, 0, 0],
                                 [0, 0.8660254, -0.5]])
    vicon_to_cam_T = np.asarray([0,0,0.3])
    vicon_to_cam_G = RT_to_SE3(vicon_to_cam_R, vicon_to_cam_T)
    

    for demo_name in demo_list:
        pose_path = pose_dir+'/'+demo_name+'/pose_traj.npy'
        output_path = output_dir+'/'+demo_name
        transform_path = transform_dir+'/'+demo_name+'/position.npy'

        pose = np.load(pose_path)
        transform = np.load(transform_path)

        traj = apply_transform(pose, transform)
        
        robot_T = transform[0]
        robot_quaternion = transform[1]
        robot_R = quaternion_to_R(robot_quaternion)
        robot_G = RT_to_SE3(robot_R,robot_T)
        robot_se3 = SE3_to_se3(robot_G)
        robot.apply_pose(robot_se3)
        robot.plot(ax, scale = 0.3, linewidth = 3)

        cam_SE3 = np.matmul(robot_G, vicon_to_cam_G)
        cam_se3 = SE3_to_se3(cam_SE3)
        cam.apply_pose(cam_se3)
        cam.plot(ax, scale = 0.2, linewidth = 3)
        ax.scatter(cam_se3[0], cam_se3[1], cam_se3[2], alpha = 0.5, s = 200)

        for obj in range(traj.shape[1]):
            #ax.plot(traj[:,obj,0], traj[:,obj,1], traj[:,obj,2], 
            #            color[obj], alpha = 0.5, linewidth = 4)
            ax.plot(traj[:,obj,0], traj[:,obj,1], traj[:,obj,2], alpha = 0.5, linewidth = 4)
            

        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')
        util.set_axes_equal(ax)
        #ax.set_zlim([0,1.2])
        #ax.axis('equal')
        #ax.set_aspect('equal')
        ax.view_init(elev = -60, azim = -90)
        fig.savefig(output_dir+'/traj_'+demo_name+'.png')

    plt.show()
    fig.savefig(output_dir+'/traj.png')    
    #IPython.embed()



'''