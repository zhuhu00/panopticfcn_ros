cmake_minimum_required(VERSION 3.0.2)
project(segment_ros)

# Check for c++14 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
IF(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  pano_msg
  message_filters
  image_transport
  tf
  pcl_catkin
  pcl_conversions
  pcl_ros
)

find_package(Boost REQUIRED COMPONENTS thread filesystem system)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      find_package(OpenCV 2.4.3 QUIET)
      if(NOT OpenCV_FOUND)
         message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
     endif()
   endif()
endif()


catkin_package(
 INCLUDE_DIRS include include/depth_segmentation
 LIBRARIES {PROJECT_NAME}
 CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs pano_msg message_filters image_transport tf
 pcl_ros pcl_conversions pcl_catkin
)


include_directories(
  include
  include/depth_segmentation
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIR}
#  ${EIGEN3_INCLUDE_DIR}
#  ${PCL_INCLUDE_DIRS}
)
add_library(${PROJECT_NAME} SHARED
  src/pc_processing.cc
  src/depth_segmentation.cc
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
add_executable(pc_seg_generator_node src/pc_seg_generator_node.cc)

add_dependencies(pc_seg_generator_node ${PROJECT_NAME})

target_link_libraries(pc_seg_generator_node
  ${catkin_LIBRARIES} ${PROJECT_NAME}
  ${OpenCV_LIBS}
)
add_executable(image_test test/test.cpp)

target_link_libraries(image_test
  ${catkin_LIBRARIES} ${PROJECT_NAME}
  ${OpenCV_LIBS}
)
