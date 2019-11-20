# setup for ros, catkin

## 1. source bashrc 
source ~/.bashrc

## 2. clean and build 
# as cv_Bridge has trouble in python3, we follow the trouble shooting
# https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3 

cd catkin_ws
catkin clean
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
catkin build
source install/setup.bash --extend
source ./devel/setup.bash
cd ..

## 3. 

