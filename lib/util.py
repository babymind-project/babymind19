import datetime
import os
import shutil
import yaml

import IPython
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

def load_env(config):
    if config['env']['type'] == 'openai':
        import gym
        env = gym.make(config['env']['name'])
    elif config['env']['type'] == 'my_env':
        from lib.env import my_env
        env = my_env.make(config['env']['name'])
    return env

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

