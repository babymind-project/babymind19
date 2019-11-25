import datetime
import os
import shutil
import yaml

import cv2
import IPython
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def create_dir(dir, clear = False):
    '''
    Check whether system has a directory of 'dir'
    If it does not exist, create it, else, empty 'dir' if clear = True.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        if clear == True:
            print('Clear directory of'+dir)
            shutil.rmtree(dir)
            os.makedirs(dir)
        elif clear == False:
            #print('Directory is already exist'+dir)
            pass

def load_yaml(yaml_dir):
    '''
    load yaml file
    '''
    with open(yaml_dir) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def load_txt(txt_path):
    '''
    load txt file
    genreate list whose element is a line of the file
    '''
    lines = []
    with open(txt_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                lines.append(line.rstrip())

    return lines

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


def load_env(config):
    if config['env']['type'] == 'openai':
        import gym
        env = gym.make(config['env']['name'])
    elif config['env']['type'] == 'my_env':
        from lib.env import my_env
        env = my_env.make(config['env']['name'])
    return env

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


class Writer():
    def __init__(self, log_file, append = False):
        self.log_file = log_file
        log_path = Path(log_file)
        log_dir = './'+str(log_path.parent)
        
        create_dir(log_dir, clear = False)
        if append:
            with open(self.log_file, 'a') as f:
                f.write(str(datetime.datetime.now())+'\n')
        else:
            with open(self.log_file, 'w') as f:
                f.write(str(datetime.datetime.now())+'\n')

    def __call__(self, string):
        with open(self.log_file, 'a') as f:
            f.write(string+'\n')

