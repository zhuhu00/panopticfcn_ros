#!/bin/bash
 
set -e

# 设置ros环境
source "/opt/ros/melodic/setup.bash"
source "/root/catkin_ws/devel/setup.bash"


# 可以对一些库进行构建和install

echo "================Panoptic_ROS Env Ready================"

cd /root/catkin_ws

exec "$@"