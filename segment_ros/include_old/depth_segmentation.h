// BSD 3-Clause License

// Copyright (c) 2019, ETHZ ASL
// All rights reserved.

#ifndef DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
#define DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_

#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>

#include "depth_common.h"
#include "opencv_macro_adapt.h"

namespace SegmentROS {

    extern int NFrame;
    extern uchar background_label;
    extern std::string output_folder;

namespace depth_segmentation {

class DepthSegmenter {
 public:
  DepthSegmenter(ros::NodeHandle& node_handle,  bool visualize_geo_seg, bool save_img);
  ~DepthSegmenter(){};

  void setCamIntrinsics(const cv::Mat& K, int width, int height);
  
  void computeDepthMap(const cv::Mat& depth_image, cv::Mat* depth_map);
  void computeDepthDiscontinuityMap(const cv::Mat& depth_image,
                                    cv::Mat* depth_discontinuity_map);
  void computeMaxDistanceMap(const cv::Mat& image, cv::Mat* max_distance_map);
  void computeNormalMap(const cv::Mat& depth_map, cv::Mat* normal_map);
  void computeMinConvexityMap(const cv::Mat& depth_map,
                              const cv::Mat& normal_map,
                              cv::Mat* min_convexity_map);
  void computeFinalEdgeMap(const cv::Mat& convexity_map,
                           const cv::Mat& distance_map,
                           const cv::Mat& discontinuity_map, cv::Mat* edge_map);
  void edgeMap(const cv::Mat& image, cv::Mat* edge_map);
  void labelMap(const cv::Mat& rgb_image,
                const cv::Mat& depth_image,
                const cv::Mat& depth_map, const cv::Mat& edge_map,
                const cv::Mat& normal_map, cv::Mat* labeled_map,
                std::vector<cv::Mat>* segment_masks, std::vector<Segment>* segments);

  void inpaintImage(const cv::Mat& depth_image, const cv::Mat& edge_map,
                    const cv::Mat& label_map, cv::Mat* inpainted);
  void findBlobs(const cv::Mat& binary,
                 std::vector<std::vector<cv::Point2i>>* labels);
  void segmentsToPC(const std::vector<Segment>& segments,
      std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds);

  void segmentSingleFrame(const cv::Mat& rgb_image, const cv::Mat& depth_image, 
      std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds);


 private:
  void generateRandomColorsAndLabels(size_t contours_size,
                                     std::vector<cv::Scalar>* colors,
                                     std::vector<int>* labels);
  cv::Mat K_;
  Params params_;

  bool if_save_img;

  cv::rgbd::RgbdNormals rgbd_normals_;
  std::vector<cv::Scalar> colors_;
  std::vector<int> labels_;
};

}  // namespace depth_segmentation

}//SegmentROS

#endif  // DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
