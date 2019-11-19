import cv2
import os
import shutil
import yaml

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from lib.math import *
def create_dir(dir, clear = False):
    '''
    check whether system has 'dir', and if not create it
    if clear==True: empty the exiting files in 'dir'
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        if clear == True:
            print('Initialize directory of'+dir)
            shutil.rmtree(dir)
            os.makedirs(dir)

def load_yaml(yaml_dir):
    '''
    load yaml file
    '''
    with open(yaml_dir) as f:
        config = yaml.load(f)
    return config

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def load_vicon(vicon_numpy_dir):
    g_load = np.load(vicon_numpy_dir, allow_pickle = True)
    g_quater = g_load[1]+g_load[2] # append list
    g_se3 = quaternion_to_se3(g_quater) 
    g_SE3 = se3_to_SE3(g_se3) 
    return g_se3, g_SE3

class frame_to_video():
    def __init__(self, img_dir, output_path, img_type='png', fps=30):
        self.img_dir = img_dir
        self.img_type = img_type
        self.output_path = output_path
        self.fps = fps
        
    def __call__(self):
        img_dir = self.img_dir
        img_type = self.img_type
        output_path = self.output_path
        fps = self.fps

        images = [img for img in sorted(os.listdir(img_dir)) if img.endswith('.'+img_type)]
        frame = cv2.imread(os.path.join(img_dir, images[0]) )
        height, width, layers = frame.shape

        video = cv2.VideoWriter(output_path, 0, fps, (width, height))

        for img in images:
            video.write(cv2.imread(os.path.join(img_dir, img)))
        #IPython.embed()

        cv2.destroyAllWindows()
        video.release()