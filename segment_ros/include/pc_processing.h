#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/common/common.h>
#include <pcl/common/impl/common.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/impl/centroid.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/impl/conditional_removal.hpp>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/impl/radius_outlier_removal.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/impl/statistical_outlier_removal.hpp>

#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/impl/lccp_segmentation.hpp>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>

#include <pcl/features/normal_3d_omp.h>

#include <pcl/common/io.h>

#include <thread>
#include <boost/format.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkPolyLine.h>
#include <chrono>

#include "common.h"
// #include "utils.h"


namespace SegmentROS
{

extern uchar background_label;
extern int NFrame;
extern std::string output_folder;


struct LCCPParam{
    // Supervoxel stuff
    float voxel_resolution = 0.008f;
    float seed_resolution = 0.05f;
    float color_importance = 0.0f;
    float spatial_importance = 1.0f;
    float normal_importance = 4.0f;
    bool use_single_cam_transform = true;
    bool use_supervoxel_refinement = true;

    // LCCPSegmentation stuff
    float concavity_tolerance_threshold = 10;
    float smoothness_threshold = 0.1;
    int min_segment_size = 3;
    bool use_extended_convexity = false;
    bool use_sanity_criterion = false;

    unsigned int k_factor = 0;

    bool show_supervoxels = false;
};

struct FilterParam{
    // Voxel grid
    // hu zhu
    // float leaf_size = 0.005f;
    float leaf_size = 0.005f;
    bool use_voxel_downsample = true;

    // Euclidean clustering
    float cluster_tolerance = 0.05f;
    int min_cluster_size = 10;
    int max_cluster_size = 25000;

    // Normal estimation
    float search_radius = 0.03f;

    //
};

struct Param{
    LCCPParam lccp_param;
    FilterParam filter_param;
};


class PCProcessor
{

public:
    PCProcessor(ros::NodeHandle& node_handle,  bool visualize_geo_seg);
    ~PCProcessor(){};

    void SetCamIntrinsics(const cv::Mat& K);

    void GeneratePC(const cv::Mat& imRGB, const cv::Mat& imDepth, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);
    void GeneratePCSem (const cv::Mat& imRGB, const cv::Mat& imDepth, const cv::Mat& imSeg, const std::vector<Obj2D>& objects, 
                                      const std::vector<Sem2D>& semantics, const std::vector<int>& mov_list,
                                      std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds);

    void FilterPCVoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filtered);

    void FilterPCEuclidean(pcl::PointCloud<PointSurfelLabel>::Ptr& cloud,
                                   pcl::PointCloud<PointSurfelLabel>::Ptr& cloud_filtered);

    void ComputeNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& normal_cloud);

    void SegmentPCGeo(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds); 

    void SegmentSingleFrame(const cv::Mat& imRGB, const cv::Mat& imDepth, std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds);

private:
    LCCPParam lccp_param_;
    FilterParam filter_param_;

    cv::Mat K_;

    bool if_visualize;
    bool if_first_time = true;

    pcl::visualization::PCLVisualizer::Ptr viewer;

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    pcl::EuclideanClusterExtraction<PointSurfelLabel> ec;
    pcl::NormalEstimationOMP <pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::LCCPSegmentation<pcl::PointXYZRGB> lccp;
    pcl::search::KdTree<PointSurfelLabel>::Ptr tree_label;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal;

};


bool IsInBB(const int x, const int y, const std::vector<cv::Rect> bbs)
{
    bool is_in_bb = false;
    for (int i = 0; i < bbs.size(); i++)
    {
        cv::Rect bb = bbs[i];
        if ((x > bb.x) && x < (bb.x + bb.width) && (y > bb.y) && (y < bb.y+bb.height))
        {
            is_in_bb = true;
            break;
        }
    }
    return is_in_bb;
}






}