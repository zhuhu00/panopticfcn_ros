#ifndef COMMON_H_
#define COMMON_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <set>

// include ros
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>

// include pcl 
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz/vizcore.hpp>

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

namespace depth_segmentation {

struct Segment {
  std::vector<cv::Vec3f> points;
  std::vector<cv::Vec3f> normals;
  std::vector<cv::Vec3f> original_colors;
  std::set<size_t> label;
  std::set<size_t> instance_label;
  std::set<size_t> semantic_label;
};

const static std::string kDebugWindowName = "DebugImages";
constexpr bool kUseTracker = false;

enum SurfaceNormalEstimationMethod {
  kFals = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS,
  kLinemod = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,
  kSri = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI,
  kDepthWindowFilter = 3,
};

struct SurfaceNormalParams {
//   SurfaceNormalParams() {
//     if(window_size % 2u!=1u)
//         std::cout<<
//     CHECK_GT(window_size, 1u);
//     if (method != SurfaceNormalEstimationMethod::kDepthWindowFilter) {
//       CHECK_LT(window_size, 8u);
//     }
//   }
  size_t window_size = 13u;
  int method =
      SurfaceNormalEstimationMethod::kDepthWindowFilter;
  bool display = false;
  double distance_factor_threshold = 0.05;
};

struct MaxDistanceMapParams {
//   MaxDistanceMapParams() { CHECK_EQ(window_size % 2u, 1u); }
  bool use_max_distance = true;
  size_t window_size = 1u;
  bool display = false;
  bool exclude_nan_as_max_distance = false;
  bool ignore_nan_coordinates = false;  // TODO(ff): This probably doesn't make
                                        // a lot of sense -> consider removing
                                        // it.
  bool use_threshold = true;
  double noise_thresholding_factor = 10.0;
  double sensor_noise_param_1st_order = 0.0012;  // From Nguyen et al. (2012)
  double sensor_noise_param_2nd_order = 0.0019;  // From Nguyen et al. (2012)
  double sensor_noise_param_3rd_order = 0.0001;  // From Nguyen et al. (2012)
  double sensor_min_distance = 0.02;
};

struct DepthDiscontinuityMapParams {
//   DepthDiscontinuityMapParams() { CHECK_EQ(kernel_size % 2u, 1u); }
  bool use_discontinuity = true;
  size_t kernel_size = 3u;
  double discontinuity_ratio = 0.01;
  bool display = false;
};

struct MinConvexityMapParams {
//   MinConvexityMapParams() { CHECK_EQ(window_size % 2u, 1u); }
  bool use_min_convexity = true;
  size_t morphological_opening_size = 1u;
  size_t window_size = 5u;
  size_t step_size = 1u;
  bool display = false;
  bool use_morphological_opening = true;
  bool use_threshold = true;
  double threshold = 0.97;
  double mask_threshold = -0.0005;
};

struct FinalEdgeMapParams {
  size_t morphological_opening_size = 1u;
  size_t morphological_closing_size = 1u;
  bool use_morphological_opening = true;
  bool use_morphological_closing = true;
  bool display = false;
};

enum LabelMapMethod {
  kFloodFill = 0,
  kContour = 1,
};

struct LabelMapParams {
  int method = LabelMapMethod::kContour;
  size_t min_size = 500u;
  bool use_inpaint = false;
  size_t inpaint_method = 0u;
  bool display = true;
};

struct SemanticInstanceSegmentationParams {
  bool enable = false;
  float overlap_threshold = 0.8f;
};

struct IsNan {
  template <class T>
  bool operator()(T const& p) const {
    return std::isnan(p);
  }
};

struct IsNotNan {
  template <class T>
  bool operator()(T const& p) const {
    return !std::isnan(p);
  }
};

struct Params {
  bool dilate_depth_image = false;
  size_t dilation_size = 1u;
  FinalEdgeMapParams final_edge;
  LabelMapParams label;
  DepthDiscontinuityMapParams depth_discontinuity;
  MaxDistanceMapParams max_distance;
  MinConvexityMapParams min_convexity;
  SurfaceNormalParams normals;
  SemanticInstanceSegmentationParams semantic_instance_segmentation;
  bool visualize_segmented_scene = false;
};

void computeCovariance(const cv::Mat& neighborhood, const cv::Vec3f& mean,
                       const size_t neighborhood_size, cv::Mat* covariance) {

  if (neighborhood.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (covariance == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (neighborhood.rows != 3u || neighborhood_size <= 0 || neighborhood_size > neighborhood.cols)
  {
    ROS_ERROR("Invalid neighborhood parameter range!");
    return;
  }

  *covariance = cv::Mat::zeros(3, 3, CV_32F);

  for (size_t i = 0u; i < neighborhood_size; ++i) {
    cv::Vec3f point;
    for (size_t row = 0u; row < neighborhood.rows; ++row) {
      point[row] = neighborhood.at<float>(row, i) - mean[row];
    }

    covariance->at<float>(0, 0) += point[0] * point[0];
    covariance->at<float>(0, 1) += point[0] * point[1];
    covariance->at<float>(0, 2) += point[0] * point[2];
    covariance->at<float>(1, 1) += point[1] * point[1];
    covariance->at<float>(1, 2) += point[1] * point[2];
    covariance->at<float>(2, 2) += point[2] * point[2];
  }
  // Assign the symmetric elements of the covariance matrix.
  covariance->at<float>(1, 0) = covariance->at<float>(0, 1);
  covariance->at<float>(2, 0) = covariance->at<float>(0, 2);
  covariance->at<float>(2, 1) = covariance->at<float>(1, 2);
}

size_t findNeighborhood(const cv::Mat& depth_map, const size_t window_size,
                        const float max_distance, const size_t x,
                        const size_t y, cv::Mat* neighborhood,
                        cv::Vec3f* mean) {
  if (depth_map.empty())
  {
    ROS_ERROR("Empty input!");
    return 0;
  }

  if (neighborhood == NULL || mean == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return 0;
  }

  if (window_size <= 0u || window_size % 2u != 1u || x >= depth_map.cols || y >= depth_map.rows)
  {
    ROS_ERROR("Invalid input parameter range!");
    return 0;
  }

  size_t neighborhood_size = 0u;
  *neighborhood = cv::Mat::zeros(3, window_size * window_size, CV_32FC1);
  cv::Vec3f mid_point = depth_map.at<cv::Vec3f>(y, x);
  for (size_t y_idx = 0u; y_idx < window_size; ++y_idx) {
    const int y_filter_idx = y + y_idx - window_size / 2u;
    if (y_filter_idx < 0 || y_filter_idx >= depth_map.rows) {
      continue;
    }
    // CHECK_GE(y_filter_idx, 0u);
    // CHECK_LT(y_filter_idx, depth_map.rows);
    for (size_t x_idx = 0u; x_idx < window_size; ++x_idx) {
      const int x_filter_idx = x + x_idx - window_size / 2u;
      if (x_filter_idx < 0 || x_filter_idx >= depth_map.cols) {
        continue;
      }
    //   CHECK_GE(x_filter_idx, 0u);
    //   CHECK_LT(x_filter_idx, depth_map.cols);

      cv::Vec3f filter_point =
          depth_map.at<cv::Vec3f>(y_filter_idx, x_filter_idx);

      // Compute Euclidean distance between filter_point and mid_point.
      const cv::Vec3f difference = mid_point - filter_point;
      const float euclidean_dist = cv::sqrt(difference.dot(difference));
      if (euclidean_dist < max_distance) {
        // Add the filter_point to neighborhood set.
        for (size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
          neighborhood->at<float>(coordinate, neighborhood_size) =
              filter_point[coordinate];
        }
        ++neighborhood_size;
        *mean += filter_point;
      }
    }
  }
//   CHECK_GE(neighborhood_size, 1u);
//   CHECK_LE(neighborhood_size, window_size * window_size);
  *mean /= static_cast<float>(neighborhood_size);
  return neighborhood_size;
}

// \brief Compute point normals of a depth image.
//
// Compute the point normals by looking at a neighborhood around each pixel.
// We're taking a standard squared kernel, where we discard points that are too
// far away from the center point (by evaluating the Euclidean distance).
//
void computeOwnNormals(const SurfaceNormalParams& params,
                       const cv::Mat& depth_map, cv::Mat* normals) {
  if (depth_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (normals == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  cv::Mat neighborhood =
      cv::Mat::zeros(3, params.window_size * params.window_size, CV_32FC1);
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::Mat covariance(3, 3, CV_32FC1);
  covariance = cv::Mat::zeros(3, 3, CV_32FC1);
  cv::Vec3f mean;
  cv::Vec3f mid_point;

  constexpr float float_nan = std::numeric_limits<float>::quiet_NaN();
#pragma omp parallel for private(neighborhood, eigenvalues, eigenvectors, \
                                 covariance, mean, mid_point)
  for (size_t y = 0u; y < depth_map.rows; ++y) {
    for (size_t x = 0u; x < depth_map.cols; ++x) {
      mid_point = depth_map.at<cv::Vec3f>(y, x);
      // Skip point if z value is nan.
      if (cvIsNaN(mid_point[0]) || cvIsNaN(mid_point[1]) ||
          cvIsNaN(mid_point[2]) || (mid_point[2] == 0.0)) {  
        normals->at<cv::Vec3f>(y, x) =
            cv::Vec3f(float_nan, float_nan, float_nan);
        continue;
      }
      const float max_distance =
          params.distance_factor_threshold * mid_point[2];
      mean = cv::Vec3f(0.0f, 0.0f, 0.0f);

      const size_t neighborhood_size =
          findNeighborhood(depth_map, params.window_size, max_distance, x, y,
                           &neighborhood, &mean);
      if (neighborhood_size > 1u) {
        computeCovariance(neighborhood, mean, neighborhood_size, &covariance);
        // Compute Eigen vectors.
        cv::eigen(covariance, eigenvalues, eigenvectors);
        // Get the Eigenvector corresponding to the smallest Eigenvalue.
        constexpr size_t n_th_eigenvector = 2u;
        for (size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
          normals->at<cv::Vec3f>(y, x)[coordinate] =
              eigenvectors.at<float>(n_th_eigenvector, coordinate);
        }
        // Re-Orient normals to point towards camera.
        if (normals->at<cv::Vec3f>(y, x)[2] > 0.0f) {
          normals->at<cv::Vec3f>(y, x) = -normals->at<cv::Vec3f>(y, x);
        }
      } else {
        normals->at<cv::Vec3f>(y, x) =
            cv::Vec3f(float_nan, float_nan, float_nan);
      }
    }
  }
}


void fillPoint(const cv::Vec3f& point, const cv::Vec3f& normals,
                 const cv::Vec3f& colors, const size_t& semantic_label,
                 const size_t& instance_label, PointSurfelLabel* point_pcl) {
    point_pcl->x = point[0];
    point_pcl->y = point[1];
    point_pcl->z = point[2];
    point_pcl->normal_x = normals[0];
    point_pcl->normal_y = normals[1];
    point_pcl->normal_z = normals[2];
    point_pcl->r = colors[0];
    point_pcl->g = colors[1];
    point_pcl->b = colors[2];

    point_pcl->semantic_label = semantic_label;
    point_pcl->instance_label = instance_label;
}


}  // namespace depth_segmentation


}//SegmentROS
#endif