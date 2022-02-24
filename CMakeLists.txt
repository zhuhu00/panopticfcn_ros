cmake_minimum_required(VERSION 3.0.2)
project(panoptic_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
geometry_msgs
nav_msgs
sensor_msgs
octomap_msgs
octomap_ros
roscpp
rospy
rosbag
std_msgs
tf
eigen_conversions
cv_bridge
image_transport
)

## System dependencies are found with CMake's conventions

find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs cv_bridge image_transport
  DEPENDS EIGEN3 PCL Ceres 
  INCLUDE_DIRS include
)

install(PROGRAMS
  scripts/pano_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
