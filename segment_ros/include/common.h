#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>

#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>

// include pcl header first to avoid building error
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>

#include <Eigen/Core>

struct PointSurfelLabel {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB;
  uint8_t instance_label;
  uint8_t semantic_label;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointSurfelLabel,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(float, rgb, rgb)(
        uint8_t, instance_label, instance_label)(uint8_t, semantic_label,
                                                 semantic_label))


namespace SegmentROS
{

struct Obj2D{
    int id;
    int category;
    float score;
    cv::Rect box; //top left x, y, width, height
    int area;
    bool is_dynamic;
    cv::Vec3b color;
    std::string cate_name;
};

struct Sem2D{
    int id;
    int category;
    int area;
    cv::Vec3b color;
    std::string cate_name;
};

struct PanoClass{
    std::string name;
    std::vector<int> category_id;
    cv::Vec3b color;
};

enum GeometricSegMode{
    kDepth = 0,
    kPointCloud = 1
};


}
#endif