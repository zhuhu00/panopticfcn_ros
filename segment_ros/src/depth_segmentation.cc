/* BSD 3-Clause License

Copyright (c) 2019, ETHZ ASL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include "utils.h"
#include "depth_segmentation.h"
#include "opencv3_macro_adapt.h"

#include <algorithm>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

namespace SegmentROS {

namespace depth_segmentation {

DepthSegmenter::DepthSegmenter(ros::NodeHandle& node_handle, bool visualize_geo_seg, bool save_img): if_save_img(save_img)
{
    // load params
    int param_holder;
    node_handle.param<bool>("depth/dilate_depth_image", params_.dilate_depth_image, false);
    node_handle.param<int>("depth/dilation_size", param_holder, 1);
    params_.dilation_size = static_cast<size_t>(param_holder);
    
    // normals
    node_handle.param<int>("depth/normals_window_size", param_holder, 13);
    params_.normals.window_size = static_cast<size_t>(param_holder);
    node_handle.param<int>("depth/normals_method", params_.normals.method, SurfaceNormalEstimationMethod::kDepthWindowFilter);
    if (params_.normals.window_size % 2u != 1u) {
        // Resetting the config value to its previous value.
        params_.normals.window_size = 13u;
        ROS_ERROR("Set the normals window size to an odd number.");
    }
    if (params_.normals.window_size < 1u) {
        // Resetting the config value to its previous value.
        params_.normals.window_size = 13u;
        ROS_ERROR("Set the normals window size to an odd value of at least 3.");
    }
    if (params_.normals.method != SurfaceNormalEstimationMethod::kDepthWindowFilter &&
        params_.normals.window_size >= 8u) {
        // Resetting the config value to its previous value.
        params_.normals.window_size = 7u;
        ROS_ERROR("Only normal method Own supports normal window sizes larger "
                    "than 7.");
    }
    node_handle.param<double>("depth/normals_distance_factor_threshold", params_.normals.distance_factor_threshold, 0.05);
    node_handle.param<bool>("depth/normals_display", params_.normals.display, false);

    // Depth discontinuity map 
    node_handle.param<int>("depth/depth_discontinuity_kernel_size", param_holder, 3);
    params_.depth_discontinuity.kernel_size = static_cast<size_t>(param_holder);
    if (params_.depth_discontinuity.kernel_size % 2u != 1u) {
        // Resetting the config value to its previous value.
        params_.depth_discontinuity.kernel_size = 3u;
        ROS_ERROR("Set the depth discontinuity kernel size to an odd number.");
    }
    node_handle.param<bool>("depth/depth_discontinuity_use_depth_discontinuity", params_.depth_discontinuity.use_discontinuity, true);
    node_handle.param<int>("depth/depth_discontinuity_kernel_size", param_holder, 3);
    params_.depth_discontinuity.use_discontinuity = static_cast<size_t>(param_holder);
    node_handle.param<double>("depth/depth_discontinuity_ratio", params_.depth_discontinuity.discontinuity_ratio, 0.01);
    node_handle.param<bool>("depth/depth_discontinuity_display", params_.depth_discontinuity.display, false);

    // Max distance map 
    node_handle.param<int>("depth/max_distance_window_size", param_holder, 1);
    params_.max_distance.window_size = static_cast<size_t>(param_holder);
    if (params_.max_distance.window_size % 2u != 1u) {
        // Resetting the config value to its previous value.
        params_.max_distance.window_size = 1u;
        ROS_ERROR("Set the max distance window size to an odd number.");
    }    
    node_handle.param<bool>("depth/max_distance_use_max_distance", params_.max_distance.use_max_distance, true);
    node_handle.param<bool>("depth/max_distance_display", params_.max_distance.display, false);
    node_handle.param<bool>("depth/max_distance_exclude_nan_as_max_distance", params_.max_distance.exclude_nan_as_max_distance, false);
    node_handle.param<bool>("depth/max_distance_ignore_nan_coordinates", params_.max_distance.ignore_nan_coordinates, false);
    node_handle.param<double>("depth/max_distance_noise_thresholding_factor", params_.max_distance.noise_thresholding_factor, 10.0);
    node_handle.param<double>("depth/max_distance_sensor_min_distancer", params_.max_distance.sensor_min_distance, 0.02);
    node_handle.param<double>("depth/max_distance_sensor_noise_param_1st_order", params_.max_distance.sensor_noise_param_1st_order, 0.0012);
    node_handle.param<double>("depth/max_distance_sensor_noise_param_2nd_order", params_.max_distance.sensor_noise_param_2nd_order, 0.0019);
    node_handle.param<double>("depth/max_distance_sensor_noise_param_3rd_order", params_.max_distance.sensor_noise_param_3rd_order, 0.0001);
    node_handle.param<bool>("depth/max_distance_use_threshold", params_.max_distance.use_threshold, true);

    // Min convexity map 
    node_handle.param<int>("depth/min_convexity_window_size", param_holder, 5);
    params_.min_convexity.window_size = static_cast<size_t>(param_holder);
    if (params_.min_convexity.window_size % 2u != 1u) {
        // Resetting the config value to its previous value.
        params_.min_convexity.window_size = 5u;
        ROS_ERROR("Set the min convexity window size to an odd number.");
    }
    node_handle.param<bool>("depth/min_convexity_use_min_convexity", params_.min_convexity.use_min_convexity, true);
    node_handle.param<int>("depth/min_convexity_morphological_opening_size", param_holder, 1);
    params_.min_convexity.morphological_opening_size = static_cast<size_t>(param_holder);
    node_handle.param<int>("depth/min_convexity_step_size", param_holder, 1);
    params_.min_convexity.step_size = static_cast<size_t>(param_holder);
    node_handle.param<bool>("depth/min_convexity_use_morphological_opening", params_.min_convexity.use_morphological_opening, true);
    node_handle.param<bool>("depth/min_convexity_use_threshold", params_.min_convexity.use_threshold, true);
    node_handle.param<double>("depth/min_convexity_threshold", params_.min_convexity.threshold, 0.97);
    node_handle.param<double>("depth/min_convexity_mask_threshold", params_.min_convexity.mask_threshold, -0.0005);
    node_handle.param<bool>("depth/min_convexity_display", params_.min_convexity.display, false);
    

    // Final edge map 
    node_handle.param<int>("depth/final_edge_morphological_opening_size", param_holder, 1);
    params_.final_edge.morphological_opening_size = static_cast<size_t>(param_holder);
    node_handle.param<int>("depth/final_edge_morphological_closing_size", param_holder, 1);
    params_.final_edge.morphological_closing_size = static_cast<size_t>(param_holder);
    node_handle.param<bool>("depth/final_edge_use_morphological_opening", params_.final_edge.use_morphological_opening, true);
    node_handle.param<bool>("depth/final_edge_use_morphological_closing", params_.final_edge.use_morphological_closing, true);
    node_handle.param<bool>("depth/final_edge_display", params_.final_edge.display, false); 

    // Label map 
    node_handle.param<int>("depth/label_method", params_.label.method, LabelMapMethod::kContour);
    node_handle.param<int>("depth/label_min_size", param_holder, 1000);
    params_.label.min_size = static_cast<size_t>(param_holder);
    node_handle.param<int>("depth/label_inpaint_method", param_holder, 0);
    params_.label.inpaint_method = static_cast<size_t>(param_holder);
    node_handle.param<bool>("depth/label_use_inpaint", params_.label.use_inpaint, false); 
    
    params_.label.display = visualize_geo_seg;

    ROS_INFO("Initialize depth_segmenter!");
}


void DepthSegmenter::setCamIntrinsics(const cv::Mat& K, int width, int height) 
{
  K.copyTo(K_);
  rgbd_normals_ = cv::rgbd::RgbdNormals(
      width, height, CV_32F, K, params_.normals.window_size,
      static_cast<int>(params_.normals.method));
}


void DepthSegmenter::computeDepthMap(const cv::Mat& depth_image,
                                     cv::Mat* depth_map) {
  if (depth_image.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (depth_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_image.type() != CV_32FC1 || depth_map->type() != CV_32FC3)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  if (K_.empty())
  {
    ROS_ERROR("Camera info not available!");
    return;
  }

  cv::rgbd::depthTo3d(depth_image, K_, *depth_map);
}

void DepthSegmenter::computeDepthDiscontinuityMap(
    const cv::Mat& depth_image, cv::Mat* depth_discontinuity_map) {
  if (depth_image.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (depth_discontinuity_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_image.type() != CV_32FC1 || depth_discontinuity_map->type() != CV_32FC1)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  constexpr size_t kMaxValue = 1u;
  constexpr double kNanThreshold = 0.0;

  cv::Size image_size(depth_image.cols, depth_image.rows);
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(params_.depth_discontinuity.kernel_size,
                               params_.depth_discontinuity.kernel_size));

  cv::Mat depth_without_nans(image_size, CV_32FC1);
  cv::threshold(depth_image, depth_without_nans, kNanThreshold, kMaxValue,
                cv::THRESH_TOZERO);

  cv::Mat dilate_image(image_size, CV_32FC1);
  cv::dilate(depth_without_nans, dilate_image, element);
  dilate_image -= depth_without_nans;

  cv::Mat erode_image(image_size, CV_32FC1);
  cv::erode(depth_without_nans, erode_image, element);
  erode_image = depth_without_nans - erode_image;

  cv::Mat max_image(image_size, CV_32FC1);
  cv::max(dilate_image, erode_image, max_image);

  cv::Mat ratio_image(image_size, CV_32FC1);
  cv::divide(max_image, depth_without_nans, ratio_image);

  cv::threshold(ratio_image, *depth_discontinuity_map,
                params_.depth_discontinuity.discontinuity_ratio, kMaxValue,
                cv::THRESH_BINARY);

  if (params_.depth_discontinuity.display) {
    static const std::string kWindowName = "DepthDiscontinuityMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(kWindowName, (*depth_discontinuity_map));
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeMaxDistanceMap(const cv::Mat& depth_map,
                                           cv::Mat* max_distance_map) {
  if (depth_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (max_distance_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_map.type() != CV_32FC3 || max_distance_map->type() != CV_32FC1)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  max_distance_map->setTo(cv::Scalar(0.0f));

  const size_t kernel_size = params_.max_distance.window_size;
  const size_t n_kernels = kernel_size * kernel_size - 1u;

  // Define the n kernels and compute the filtered images.
  for (size_t i = 0u; i < n_kernels + 1u; ++i) {
    if (i == n_kernels / 2u) {
      continue;
    }
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    kernel.at<float>(i) = -1.0f;
    kernel.at<float>(n_kernels / 2u) = 1.0f;

    // Compute the filtered images.
    cv::Mat filtered_image(depth_map.size(), CV_32FC3);
    cv::filter2D(depth_map, filtered_image, CV_32FC3, kernel);

    // Calculate the norm over the three channels.
    std::vector<cv::Mat> channels(3);
    cv::split(filtered_image, channels);
    cv::Mat distance_map(depth_map.size(), CV_32FC1);
    if (params_.max_distance.ignore_nan_coordinates) {
      // Ignore nan values for the distance calculation.
      cv::Mat mask_0 = cv::Mat(channels[0] == channels[0]);
      cv::Mat mask_1 = cv::Mat(channels[1] == channels[1]);
      cv::Mat mask_2 = cv::Mat(channels[2] == channels[2]);
      mask_0.convertTo(mask_0, CV_32FC1);
      mask_1.convertTo(mask_1, CV_32FC1);
      mask_2.convertTo(mask_2, CV_32FC1);
      distance_map = mask_0.mul(channels[0].mul(channels[0])) +
                     mask_1.mul(channels[1].mul(channels[1])) +
                     mask_2.mul(channels[2].mul(channels[2]));
    } else {
      // If at least one of the coordinates is nan the distance will be nan.
      distance_map = channels[0].mul(channels[0]) +
                     channels[1].mul(channels[1]) +
                     channels[2].mul(channels[2]);
    }

    if (params_.max_distance.exclude_nan_as_max_distance) {
      cv::Mat mask = cv::Mat(distance_map == distance_map);
      mask.convertTo(mask, CV_32FC1);
      distance_map = mask.mul(distance_map);
    }
    // Individually set the maximum pixel value of the two matrices.
    cv::max(*max_distance_map, distance_map, *max_distance_map);
  }

  cv::sqrt(*max_distance_map, *max_distance_map);
  std::vector<cv::Mat> channels(3);
  cv::split(depth_map, channels);

  // Threshold the max_distance_map to get an edge map.
  if (params_.max_distance.use_threshold) {
    for (size_t i = 0u; i < depth_map.cols * depth_map.rows; ++i) {
      // Threshold the distance map based on Nguyen et al. (2012) noise model.
      // TODO(ff): Theta should be the angle between the normal and the camera
      // direction. (Here, a mean value is used, as suggested by Tateno et al.
      // (2016))
      static constexpr float theta = 30.f * CV_PI / 180.f;
      float z = (channels[2]).at<float>(i);
      float sigma_axial_noise =
          params_.max_distance.sensor_noise_param_1st_order +
          params_.max_distance.sensor_noise_param_2nd_order *
              (z - params_.max_distance.sensor_min_distance) *
              (z - params_.max_distance.sensor_min_distance) +
          params_.max_distance.sensor_noise_param_3rd_order / cv::sqrt(z) *
              theta * theta / (CV_PI / 2.0f - theta) * (CV_PI / 2.0f - theta);
      if (max_distance_map->at<float>(i) >
          sigma_axial_noise * params_.max_distance.noise_thresholding_factor) {
        max_distance_map->at<float>(i) = 1.0f;
      } else {
        max_distance_map->at<float>(i) = 0.0f;
      }
    }
  }
  if (params_.max_distance.display) {
    static const std::string kWindowName = "MaxDistanceMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(kWindowName, (*max_distance_map));
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeNormalMap(const cv::Mat& depth_map, cv::Mat* normal_map) {
  if (depth_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (normal_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (!(depth_map.type() == CV_32FC3 &&
            (params_.normals.method == SurfaceNormalEstimationMethod::kFals ||
             params_.normals.method == SurfaceNormalEstimationMethod::kSri ||
             params_.normals.method ==
                 SurfaceNormalEstimationMethod::kDepthWindowFilter) ||
        (depth_map.type() == CV_32FC1 || depth_map.type() == CV_16UC1 ||
         depth_map.type() == CV_32FC3) &&
            params_.normals.method == SurfaceNormalEstimationMethod::kLinemod))
  {
    ROS_ERROR("Invalid image type!");
    return;
  }

  if (params_.normals.method !=
      SurfaceNormalEstimationMethod::kDepthWindowFilter) {
    rgbd_normals_(depth_map, *normal_map);
  } else {
    computeOwnNormals(params_.normals, depth_map, normal_map);
  }
  if (params_.normals.display) {
    static const std::string kWindowName = "NormalMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    // Taking the negative values of the normal map, as all normals point in
    // negative z-direction.
    ROS_INFO("Display normal!");        
    cv::imshow(kWindowName, (-(*normal_map)));
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeMinConvexityMap(const cv::Mat& depth_map,
                                            const cv::Mat& normal_map,
                                            cv::Mat* min_convexity_map) {
  if (depth_map.empty() || normal_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (min_convexity_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_map.type() != CV_32FC3 || normal_map.type() != CV_32FC3 || min_convexity_map->type() != CV_32FC1)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }
  
  min_convexity_map->setTo(cv::Scalar(10.0f));

  const size_t kernel_size = params_.min_convexity.window_size +
                             (params_.min_convexity.step_size - 1u) *
                                 (params_.min_convexity.window_size - 1u);
  const size_t n_kernels =
      params_.min_convexity.window_size * params_.min_convexity.window_size -
      1u;
  // Define the n point-wise distance kernels and compute the filtered images.
  // The kernels for i look as follows (e.g. window_size = 5, i = 6):
  //     0  0  0  0  0
  //     0  1  0  0  0
  //     0  0 -1  0  0
  //     0  0  0  0  0
  //     0  0  0  0  0
  for (size_t i = 0u; i < n_kernels + 1u;
       i += static_cast<size_t>(i % kernel_size == kernel_size) * kernel_size +
            params_.min_convexity.step_size) {
    if (i == n_kernels / 2u) {
      continue;
    }
    cv::Mat difference_kernel =
        cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    difference_kernel.at<float>(i) = 1.0f;
    difference_kernel.at<float>(n_kernels / 2u) = -1.0f;

    // Compute the filtered images.
    cv::Mat difference_map(depth_map.size(), CV_32FC3);
    cv::filter2D(depth_map, difference_map, CV_32FC3, difference_kernel);

    // Calculate the dot product over the three channels of difference_map and
    // normal_map.
    cv::Mat difference_times_normal(depth_map.size(), CV_32FC3);
    // cv::Mat neg_normal_map = -normal_map; // Work round for opencv3.4
    difference_times_normal = -(difference_map.mul(normal_map));
    std::vector<cv::Mat> channels(3);
    cv::split(difference_times_normal, channels);
    cv::Mat vector_projection(depth_map.size(), CV_32FC1);
    vector_projection = channels[0] + channels[1] + channels[2];

    // TODO(ff): Check if params_.min_convexity.mask_threshold should be
    // mid-point distance dependent.
    // maybe do something like:
    // std::vector<cv::Mat> depth_map_channels(3);
    // cv::split(depth_map, depth_map_channels);
    // vector_projection = vector_projection.mul(depth_map_channels[2]);

    cv::Mat concavity_mask(depth_map.size(), CV_32FC1);
    cv::Mat convexity_mask(depth_map.size(), CV_32FC1);

    // Split the projected vector images into convex and concave
    // regions/masks.
    constexpr float kMaxBinaryValue = 1.0f;
    cv::threshold(vector_projection, convexity_mask,
                  params_.min_convexity.mask_threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY);
    cv::threshold(vector_projection, concavity_mask,
                  params_.min_convexity.mask_threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY_INV);

    cv::Mat normal_kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    normal_kernel.at<float>(i) = 1.0f;

    cv::Mat filtered_normal_image = cv::Mat::zeros(normal_map.size(), CV_32FC3);
    cv::filter2D(normal_map, filtered_normal_image, CV_32FC3, normal_kernel);

    cv::Mat nan_mask = (filtered_normal_image != filtered_normal_image);
    normal_map.copyTo(filtered_normal_image, nan_mask);

    // TODO(ff): Create a function for this mulitplication and projections.
    cv::Mat normal_times_filtered_normal(depth_map.size(), CV_32FC3);
    normal_times_filtered_normal = normal_map.mul(filtered_normal_image);

    nan_mask = (filtered_normal_image != filtered_normal_image);
    filtered_normal_image.copyTo(normal_times_filtered_normal, nan_mask);
    std::vector<cv::Mat> normal_channels(3);
    cv::split(normal_times_filtered_normal, normal_channels);
    cv::Mat normal_vector_projection(depth_map.size(), CV_32FC1);
    normal_vector_projection =
        normal_channels[0] + normal_channels[1] + normal_channels[2];
    normal_vector_projection = concavity_mask.mul(normal_vector_projection);

    cv::Mat convexity_map = cv::Mat::ones(depth_map.size(), CV_32FC1);
    convexity_map = convexity_mask + normal_vector_projection;

    // Individually set the minimum pixel value of the two matrices.
    cv::min(*min_convexity_map, convexity_map, *min_convexity_map);
  }

  if (params_.min_convexity.use_threshold) {
    constexpr float kMaxBinaryValue = 1.0f;
    cv::threshold(*min_convexity_map, *min_convexity_map,
                  params_.min_convexity.threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY);
  }

  if (params_.min_convexity.use_morphological_opening) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.min_convexity.morphological_opening_size + 1u,
                 2u * params_.min_convexity.morphological_opening_size + 1u),
        cv::Point(params_.min_convexity.morphological_opening_size,
                  params_.min_convexity.morphological_opening_size));
    cv::morphologyEx(*min_convexity_map, *min_convexity_map, cv::MORPH_OPEN,
                     element);
  }

  if (params_.min_convexity.display) {
    static const std::string kWindowName = "MinConcavityMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(kWindowName, (*min_convexity_map));
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeFinalEdgeMap(const cv::Mat& convexity_map,
                                         const cv::Mat& distance_map,
                                         const cv::Mat& discontinuity_map,
                                         cv::Mat* edge_map) {
  if (convexity_map.empty() || distance_map.empty() || discontinuity_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (edge_map == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (convexity_map.type() != CV_32FC1 || distance_map.type() != CV_32FC1 || discontinuity_map.type() != CV_32FC1)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }


  if (params_.final_edge.use_morphological_opening) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.final_edge.morphological_opening_size + 1u,
                 2u * params_.final_edge.morphological_opening_size + 1u),
        cv::Point(params_.final_edge.morphological_opening_size,
                  params_.final_edge.morphological_opening_size));

    cv::morphologyEx(convexity_map, convexity_map, cv::MORPH_OPEN, element);
  }
  if (params_.final_edge.use_morphological_closing) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.final_edge.morphological_closing_size + 1u,
                 2u * params_.final_edge.morphological_closing_size + 1u),
        cv::Point(params_.final_edge.morphological_closing_size,
                  params_.final_edge.morphological_closing_size));
    cv::morphologyEx(distance_map, distance_map, cv::MORPH_CLOSE, element);

    // TODO(ntonci): Consider making a separate parameter for discontinuity_map.
    cv::morphologyEx(discontinuity_map, discontinuity_map, cv::MORPH_CLOSE,
                     element);
  }
  cv::Mat distance_discontinuity_map(cv::Size(distance_map.cols, distance_map.rows), CV_32FC1);
  cv::Mat add = distance_map + discontinuity_map;
  cv::threshold(add, distance_discontinuity_map,
                1.0, 1.0, cv::THRESH_TRUNC);
  *edge_map = convexity_map - distance_discontinuity_map;
    
  // TODO(ff): Perform morphological operations (also) on edge_map.
  if (params_.final_edge.display) {
    static const std::string kWindowName = "FinalEdgeMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    imshow(kWindowName, (*edge_map));
    cv::waitKey(1);
  }
}

void DepthSegmenter::findBlobs(const cv::Mat& binary,
                               std::vector<std::vector<cv::Point2i>>* labels) {

  if (binary.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (labels == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (binary.type() != CV_32FC1)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  labels->clear();

  cv::Mat label_image;
  binary.convertTo(label_image, CV_32SC1);

  // Labels start at 2 as we use 0 for background and 1 for unlabled.
  size_t label_count = 2u;
  for (size_t y = 0u; y < label_image.rows; ++y) {
    for (size_t x = 0u; x < label_image.cols; ++x) {
      if (label_image.at<int>(y, x) != 1) {
        continue;
      }
      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0,
                    cv::FLOODFILL_FIXED_RANGE);
      std::vector<cv::Point2i> blob;
      size_t rect_size_y = rect.y + rect.height;
      size_t rect_size_x = rect.x + rect.width;
      for (size_t i = rect.y; i < rect_size_y; ++i) {
        for (size_t j = rect.x; j < rect_size_x; ++j) {
          if (label_image.at<int>(i, j) != label_count) {
            continue;
          }
          blob.push_back(cv::Point2i(j, i));
        }
      }
      if (blob.size() > 1u) {
        labels->push_back(blob);
        ++label_count;
      }
    }
  }
}

void DepthSegmenter::inpaintImage(const cv::Mat& depth_image,
                                  const cv::Mat& edge_map,
                                  const cv::Mat& label_map,
                                  cv::Mat* inpainted) {
  if (depth_image.empty() || edge_map.empty() || label_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (inpainted == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_image.type() != CV_32FC1 || edge_map.type() != CV_32FC1 || label_map.type() != CV_8UC3)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  cv::Mat gray_edge;
  cv::cvtColor(label_map, gray_edge, CV_BGR2GRAY);

  cv::Mat mask = cv::Mat::zeros(edge_map.size(), CV_8UC1);
  // We set the mask to 1 where we have depth values but no label.
  cv::Mat depth_input = depth_image == depth_image;
  cv::Mat gray_input = gray_edge == 0;
  cv::bitwise_and(depth_input, gray_input, mask);
  constexpr double kInpaintRadius = 1.0;
  cv::inpaint(label_map, mask, *inpainted, kInpaintRadius,
              params_.label.inpaint_method);
}

void DepthSegmenter::generateRandomColorsAndLabels(
    size_t contours_size, std::vector<cv::Scalar>* colors,
    std::vector<int>* labels) {
  if (colors == NULL || labels == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (contours_size < 0u)
  {
    ROS_ERROR("Contours size should be positive!");
    return;
  }

  colors->clear();
  labels->clear();

  if (colors_.size() < contours_size) {
    colors_.reserve(contours_size);
    for (size_t i = colors_.size(); i < contours_size; ++i) {
      colors_.push_back(
          cv::Scalar(255 * (rand() / static_cast<float>(RAND_MAX)),
                     255 * (rand() / static_cast<float>(RAND_MAX)),
                     255 * (rand() / static_cast<float>(RAND_MAX))));
      labels_.push_back(i);
    }
  }
  *colors = colors_;
  *labels = labels_;
}

void DepthSegmenter::labelMap(const cv::Mat& rgb_image,
                              const cv::Mat& depth_image,
                              const cv::Mat& depth_map, const cv::Mat& edge_map,
                              const cv::Mat& normal_map, cv::Mat* labeled_map,
                              std::vector<cv::Mat>* segment_masks,
                              std::vector<Segment>* segments) {

  if (rgb_image.empty() || depth_image.empty() || edge_map.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }

  if (labeled_map == NULL || segment_masks == NULL || segments == NULL)
  {
    ROS_ERROR("Null input pointer!");
    return;
  }

  if (depth_image.type() != CV_32FC1 || edge_map.type() != CV_32FC1 || depth_map.type() != CV_32FC3
        || normal_map.type() != CV_32FC3)
  {
    ROS_ERROR("Invalid image type!");
    return; 
  }

  segments->clear();

  constexpr size_t kMaskValue = 255u;

  cv::Mat original_depth_map;
  cv::rgbd::depthTo3d(depth_image, K_,
                      original_depth_map);

  cv::Mat output = cv::Mat::zeros(depth_image.size(), CV_8UC3);

  switch (params_.label.method) {
    case LabelMapMethod::kContour: {
      // TODO(ff): Move to method.
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::Mat edge_map_8u;
      edge_map.convertTo(edge_map_8u, CV_8U);
      static const cv::Point kContourOffset = cv::Point(0, 0);
      cv::findContours(edge_map_8u, contours, hierarchy,
                       cv::RETR_TREE, /*cv::RETR_CCOMP*/
                       CV_CHAIN_APPROX_NONE, kContourOffset);

      std::vector<cv::Scalar> colors;
      std::vector<int> labels;
      generateRandomColorsAndLabels(contours.size(), &colors, &labels);
      for (size_t i = 0u; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        constexpr int kNoParentContour = -1;
        if (area < params_.label.min_size) {
          const int parent_contour = hierarchy[i][3];
          if (parent_contour == kNoParentContour) {
            // Assign black color to areas that have no parent contour.
            colors[i] = cv::Scalar(0, 0, 0);
            labels[i] = -1;
            drawContours(edge_map_8u, contours, i, cv::Scalar(0u), CV_FILLED, 8,
                         hierarchy);
          } else {
            if (hierarchy[i][0] == -1 && hierarchy[i][1] == -1) {
              // Assign the color of the parent contour.
              colors[i] = colors[parent_contour];
              labels[i] = labels[parent_contour];
            } else {
              colors[i] = cv::Scalar(0, 0, 0);
              labels[i] = -1;
              drawContours(edge_map_8u, contours, i, cv::Scalar(0u), CV_FILLED,
                           8, hierarchy);
            }
          }
        }
      }

      cv::Mat output_labels =
          cv::Mat(depth_image.size(), CV_32SC1, cv::Scalar(0));
      for (size_t i = 0u; i < contours.size(); ++i) {
        drawContours(output, contours, i, cv::Scalar(colors[i]), CV_FILLED, 8,
                     hierarchy);
        drawContours(output_labels, contours, i, cv::Scalar(labels[i]),
                     CV_FILLED, 8, hierarchy);
        drawContours(edge_map_8u, contours, i, cv::Scalar(0u), 1, 8, hierarchy);
      }

      cv::Mat mask = edge_map_8u == 0u;
      output.setTo(cv::Scalar(0, 0, 0), mask);
      output_labels.setTo(-1, mask);
      // Create a map of all the labels.
      std::map<size_t, size_t> labels_map;
      size_t value = 0u;
      for (size_t i = 0u; i < labels.size(); ++i) {
        if (labels[i] >= 0) {
          // Create a new map if label is not yet in keys.
          if (labels_map.find(labels[i]) == labels_map.end()) {
            labels_map[labels[i]] = value;
            ++value;
          }
        }
      }
      segments->resize(labels_map.size());
      segment_masks->resize(labels_map.size());
      for (cv::Mat& segment_mask : *segment_masks) {
        segment_mask = cv::Mat(depth_image.size(), CV_8UC1, cv::Scalar(0));
      }
      for (size_t x = 0u; x < output_labels.cols; ++x) {
        for (size_t y = 0u; y < output_labels.rows; ++y) {
          int32_t label = output_labels.at<int32_t>(y, x);
          // Check if edge point and assign the nearest neighbor label.
          const bool is_edge_point = edge_map_8u.at<uint8_t>(y, x) == 0u &&
                                     depth_image.at<float>(y, x) > 0.0f;
          if (is_edge_point) {
            // We assign edgepoints by default to -1.
            label = -1;
            const cv::Vec3f& edge_point = depth_map.at<cv::Vec3f>(y, x);
            constexpr double kMinNearestNeighborDistance = 0.05;
            double min_dist = kMinNearestNeighborDistance;
            constexpr int kFilterSizeHalfFloored = 4u;
            for (int i = -kFilterSizeHalfFloored; i <= kFilterSizeHalfFloored;
                 ++i) {
              if (static_cast<int>(x) + i < 0) {
                continue;
              }
              if (static_cast<int>(x) + i >= output_labels.cols) {
                break;
              }
              for (int j = -kFilterSizeHalfFloored; j <= kFilterSizeHalfFloored;
                   ++j) {
                if (static_cast<int>(y) + j < 0 || (i == 0 && j == 0)) {
                  continue;
                }
                if (static_cast<int>(y) + j >= output_labels.rows) {
                  break;
                }
                const cv::Vec3f filter_point =
                    depth_map.at<cv::Vec3f>(y + j, x + i);
                const double dist = cv::norm(edge_point - filter_point);
                if (dist >= min_dist) {
                  continue;
                }
                const bool filter_point_is_edge_point =
                    edge_map_8u.at<uint8_t>(y + j, x + i) == 0u &&
                    depth_image.at<float>(y + j, x + i) > 0.0f;
                if (!filter_point_is_edge_point) {
                  const int label_tmp = output_labels.at<int32_t>(y + j, x + i);
                  if (label_tmp < 0) {
                    continue;
                  }
                  min_dist = dist;
                  label = label_tmp;
                  output_labels.at<int32_t>(y, x) = label;
                }
              }
            }
            if (label > 0) {
              output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                  colors[label][0], colors[label][1], colors[label][2]);
            }
          }
          if (label < 0) {
            continue;
          } else {
            // Append vectors from depth_map and normals from normal_map to
            // vectors of segments.
            cv::Vec3f point = original_depth_map.at<cv::Vec3f>(y, x);
            cv::Vec3f normal = normal_map.at<cv::Vec3f>(y, x);
            cv::Vec3b original_color = rgb_image.at<cv::Vec3b>(y, x);
            cv::Vec3f color_f;
            constexpr bool kUseOriginalColors = true;
            if (kUseOriginalColors) {
              color_f = cv::Vec3f(static_cast<float>(original_color[0]),
                                  static_cast<float>(original_color[1]),
                                  static_cast<float>(original_color[2]));
            } else {
              color_f = cv::Vec3f(static_cast<float>(colors[label][0]),
                                  static_cast<float>(colors[label][1]),
                                  static_cast<float>(colors[label][2]));
            }
            std::vector<cv::Vec3f> rgb_point_with_normals{point, normal,
                                                          color_f};
            Segment& segment = (*segments)[labels_map.at(label)];
            segment.points.push_back(point);
            segment.normals.push_back(normal);
            segment.original_colors.push_back(color_f);
            segment.label.insert(label);
            cv::Mat& segment_mask = (*segment_masks)[labels_map.at(label)];
            segment_mask.at<uint8_t>(y, x) = kMaskValue;
          }
        }
      }
      if (segments->size()!=labels_map.size())
        ROS_ERROR("Segment size and label size doesn't match");
      break;
    }
    case LabelMapMethod::kFloodFill: {
      // TODO(ff): Move to method.
      cv::Mat binary_edge_map;
      constexpr float kEdgeMapThresholdValue = 0.0f;
      constexpr float kMaxBinaryValue = 1.0f;
      cv::threshold(edge_map, binary_edge_map, kEdgeMapThresholdValue,
                    kMaxBinaryValue, cv::THRESH_BINARY);
      std::vector<std::vector<cv::Point2i>> labeled_segments;
      findBlobs(binary_edge_map, &labeled_segments);

      std::vector<cv::Scalar> colors;
      std::vector<int> labels;
      generateRandomColorsAndLabels(labeled_segments.size(), &colors, &labels);
      segments->resize(labeled_segments.size());
      // Assign the colors and labels to the segments.
      for (size_t i = 0u; i < labeled_segments.size(); ++i) {
        cv::Vec3b color;
        if (labeled_segments[i].size() < params_.label.min_size) {
          color = cv::Vec3b(0, 0, 0);
        } else {
          color = cv::Vec3b(colors[i][0], colors[i][1], colors[i][2]);
        }
        cv::Mat segment_mask =
            cv::Mat(depth_image.size(), CV_8UC1, cv::Scalar(0));
        for (size_t j = 0u; j < labeled_segments[i].size(); ++j) {
          const size_t x = labeled_segments[i][j].x;
          const size_t y = labeled_segments[i][j].y;
          output.at<cv::Vec3b>(y, x) = color;
          // TODO(ff): We might need this here.
          // if (color == cv::Vec3b(0, 0, 0)) {
          //   continue;
          // }
          cv::Vec3f point = depth_map.at<cv::Vec3f>(y, x);
          cv::Vec3f normal = normal_map.at<cv::Vec3f>(y, x);
          cv::Vec3b original_color = rgb_image.at<cv::Vec3f>(y, x);
          cv::Vec3f color_f{float(original_color[0]), float(original_color[1]),
                            float(original_color[2])};
          Segment& segment = (*segments)[i];
          segment.points.push_back(point);
          segment.normals.push_back(normal);
          segment.original_colors.push_back(color_f);
          segment.label.insert(i);
          segment_mask.at<uint8_t>(y, x) = kMaskValue;
        }
        segment_masks->push_back(segment_mask.clone());
      }
      break;
    }
  }

  // Remove small segments from segments vector.
  for (size_t i = 0u; i < segments->size();) {
    if ((*segments)[i].points.size() < params_.label.min_size) {
      segments->erase(segments->begin() + i);
      segment_masks->erase(segment_masks->begin() + i);
    } else {
      ++i;
    }
  }

  if (params_.label.use_inpaint) {
    inpaintImage(depth_image, edge_map, output, &output);
  }

  if (params_.label.display) {
    static const std::string kWindowName = "DepthSeg";
    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    // cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    
    cv::resizeWindow(kWindowName, 480, 360);
    imshow(kWindowName, output);
    cv::waitKey(1);
  }

  if (if_save_img)
  {
      cv::Mat geo_seg_save = output.clone();
      makePath(output_folder + "/image/geo_seg/", 0777);
      cv::imwrite(output_folder + "/image/geo_seg/" + std::to_string(NFrame)+".jpg",geo_seg_save);
  }
    
  *labeled_map = output;
}


void DepthSegmenter::segmentsToPC(const std::vector<Segment>& segments, std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds) {
    
    clouds.clear();
    for (Segment segment : segments) {
        pcl::PointCloud<PointSurfelLabel>::Ptr segment_pcl(new pcl::PointCloud<PointSurfelLabel>);
        for (std::size_t i = 0u; i < segment.points.size(); ++i) {
            PointSurfelLabel point_pcl;
            uint8_t semantic_label = background_label;
            uint8_t instance_label = 0u;

            fillPoint(segment.points[i], segment.normals[i], segment.original_colors[i],
                 semantic_label, instance_label, &point_pcl);
            segment_pcl->push_back(point_pcl);
        }
        clouds.push_back(segment_pcl);
    }
}


void DepthSegmenter::segmentSingleFrame(const cv::Mat& rgb_image, const cv::Mat& depth_image, 
        std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds) {
  if (rgb_image.empty() || depth_image.empty())
  {
    ROS_ERROR("Empty input!");
    return;
  }
  
  cv::Mat rescaled_depth = cv::Mat(depth_image.size(), CV_32FC1);
  if (depth_image.type() == CV_16UC1) {
    cv::rgbd::rescaleDepth(depth_image, CV_32FC1, rescaled_depth);
  } else if (depth_image.type() != CV_32FC1) {
    ROS_ERROR("Depth image is of unknown type");
  } else {
    rescaled_depth = depth_image;
  }

  constexpr double kZeroValue = 0.0;
  cv::Mat nan_mask = rescaled_depth != rescaled_depth;
  rescaled_depth.setTo(kZeroValue, nan_mask);

  // Compute depth map from rescaled depth image.
  cv::Mat depth_map(rescaled_depth.size(), CV_32FC3);
  computeDepthMap(rescaled_depth, &depth_map);

  // Compute normals based on specified method.
  cv::Mat normal_map(rescaled_depth.size(), CV_32FC3);
  if (params_.normals.method ==
          SurfaceNormalEstimationMethod::kFals ||
      params_.normals.method ==
          SurfaceNormalEstimationMethod::kSri ||
      params_.normals.method ==
          SurfaceNormalEstimationMethod::
              kDepthWindowFilter) {
    computeNormalMap(depth_map, &normal_map);
  } else if (params_.normals.method ==
             SurfaceNormalEstimationMethod::kLinemod) {
    computeNormalMap(depth_image, &normal_map);
  }

  // Compute depth discontinuity map.
  cv::Mat discontinuity_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
  if (params_.depth_discontinuity.use_discontinuity) {
    computeDepthDiscontinuityMap(rescaled_depth, &discontinuity_map);
  }

  // Compute maximum distance map.
  cv::Mat distance_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
  if (params_.max_distance.use_max_distance) {
    computeMaxDistanceMap(depth_map, &distance_map);
  }

  // Compute minimum convexity map.
  cv::Mat convexity_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
  if (params_.min_convexity.use_min_convexity) {
    computeMinConvexityMap(depth_map, normal_map, &convexity_map);
  }

  // Compute final edge map.
  cv::Mat edge_map(rescaled_depth.size(), CV_32FC1);
  computeFinalEdgeMap(convexity_map, distance_map, discontinuity_map, &edge_map);

  // Label the remaning segments.
  cv::Mat *label_map = new cv::Mat(rescaled_depth.size(), CV_32FC1);
  std::vector<cv::Mat> *segment_masks = new std::vector<cv::Mat>;
  std::vector<Segment> *segments = new std::vector<Segment>;
  cv::Mat remove_no_values = cv::Mat::zeros(edge_map.size(), edge_map.type());

  nan_mask = (rescaled_depth == rescaled_depth);
  edge_map.copyTo(remove_no_values, nan_mask);
  edge_map = remove_no_values;

  labelMap(rgb_image, rescaled_depth, depth_map, edge_map,
                           normal_map, label_map, segment_masks, segments);

  segmentsToPC(*segments, clouds);
  
  delete label_map;
  delete segment_masks;
  delete segments;
}

}  
}
