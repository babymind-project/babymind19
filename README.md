## Prerequisite
1. nvidia driver +CUDA 10.0 + CUDNN7.5
2. zed, pyzed:\
 https://github.com/stereolabs/zed-python-api \
 https://www.stereolabs.com/developers \
3. ros-melodic
4. tensorflow-gpu

## Preliminaries
### 1. Set-up vicon IP
```
gedit ./catkin_ws/vicon_bridge_launchvicon.launch
edit datastream_hostport value as yout vicon IP address
```
### 2. Set-up vicon object
vicon objects should be named same as defined in ./configure/[task_name]_objects.txt


## Main
### 0. Set-up project
```
source ./setup.sh
```

### 1. Collect data  
To watch the field of view of the camera, following code will show an rviz map of the camera.
```
python3 ./collect_data.py [task_name] [demo_name] --watch
```

If you want to record both vicon and vision, 
```
python3 ./collect_data.py [task_name] [demo_name] --vicon
```

If you want to record vision only
```
python3 ./collect_data.py [task_name] [demo_name]
```

!! Be sure all hardware collectly generates ros topics (/zed/zed_node/rgb, /vicon/k_xxx/k_xxx, ...)
```
rostopic list
rostopic echo xxx
```

Check **"raw.bag.activate"** is generated in **"./data/[task_name]/[demo_name]"**.

### 2. convert bag file to rgb/depth/vcion data
```
python3 ./read_bag.py [task_name]
```
Check **depth, rgb, vicon folders** are generated in **"./data/[task_name]/[demo_name]"**


### 3. make segment label
```
python3 ./make_segment_label.py [task_name] [demo_name]
```

### 4.

