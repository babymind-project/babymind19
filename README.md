## Prerequisite
1. nvidia driver +CUDA 10.0 + CUDNN7.5
2. zed, pyzed:\
 https://github.com/stereolabs/zed-python-api \
 https://www.stereolabs.com/developers \
3. ros-melodic
4. tensorflow-gpu

## Preliminaries
1. Setup vicon IP
```
gedit ./catkin_ws/vicon_bridge_launchvicon.launch
edit datastream_hostport value as yout vicon IP address
```

## Main
### 0. Setup project
```
source ./setup.sh
```

### 1. Collect data  
To watch the field of view of the camera, following code will show an rviz map of the camera.
```
python3 ./collect_data.py [task_name] [demo_name] --watch
```

If you want to record both vicon and vision, following code generates **"raw.bag.activate"** at **"./data/[task_name]/[demo_name]"**.
```
python3 ./collect_data.py [task_name] [demo_name] --vicon
```

If you want to record vision only
```
python3 ./collect_data.py [task_name] [demo_name]
```

### 2. Preprocess data
