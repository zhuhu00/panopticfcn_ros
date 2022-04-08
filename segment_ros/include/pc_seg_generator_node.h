#ifndef PC_SEG_GENERATOR_NODE_H_
#define PC_SEG_GENERATOR_NODE_H_

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
// #include <tf/transform_broadcaster.h>
// #include <tf/transform_listener.h>

#include "common.h"
#include "pc_processing.h"
#include "depth_segmentation/depth_segmentation.h"
#include <seg_msgs/Seg.h>

namespace SegmentROS
{
    
class PCSegGeneratorNode
{

public:
    PCSegGeneratorNode(ros::NodeHandle &node_handle);
    ~PCSegGeneratorNode();

    // image callback to generate point cloud segments
    void ImageCallback (const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
    void ImageSegCallback (const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD,const seg_msgs::SegConstPtr& msgSeg);

    // camera info callback to initialize camera intrinsics
    void CamInfoCallback (const sensor_msgs::CameraInfoConstPtr& msgInfo);
    void SetCamInfo();

    // process messages
    void ProcessImageMsg (const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD);
    void ProcessSegMsg (const seg_msgs::SegConstPtr& msgSeg);
    
    void Update();
    void LabelPC();

    cv::Mat DrawPanoSeg();
    cv::Mat DrawPanoSegwithmask();
    int GetMappedSemanticLabel(int label_in);


private:
    // image subscibers
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, seg_msgs::Seg> seg_sync_pol;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Subscriber<sensor_msgs::Image> *rgb_subscriber_;
    message_filters::Subscriber<sensor_msgs::Image> *depth_subscriber_;
    message_filters::Subscriber<seg_msgs::Seg> *seg_subscriber_;
    message_filters::Synchronizer<sync_pol> *sync_;
    message_filters::Synchronizer<seg_sync_pol> *seg_sync_;     

    // camera info subscriber
    ros::Subscriber info_sub_;

    // operation modes
    bool use_semantic_segmentation = false;
    bool use_geometric_segmentation = false;
    bool use_distance_check = false;
    bool use_direct_fusion = false;
    bool use_GT_camera_frame = true;
    std::string camera_frame;
    int geo_seg_mode = GeometricSegMode::kPointCloud;
    bool visualize_geo_seg = false;
    bool visualize_pano_seg = false;
    bool visualize_fusion_seg = false;
    bool visualize_rgb_raw = false;
    bool pub_seg_img = false;
    bool save_img = false;

    // Camera intrinsics
    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    int width;
    int height;
    bool camera_info_ready = false;

    // Frame time and id
    std::string camera_frame_id_;
    ros::Time current_frame_time_ = ros::Time::now();

    // Images to process
    cv::Mat imRGB;
    cv::Mat imDepth;
    cv::Mat imSeg;

    // Panoptic seg info
    std::unordered_map<int, int> class_id_mapping; // we use a single class id for one semantic class (as string)
    std::vector<PanoClass> pano_class_lib;
    std::vector<int> obj_class_id;
    std::vector<std::string> dyn_obj_class;

    // Detection results
    std::vector<Obj2D> objects;
    std::vector<Sem2D> semantics; 
    std::vector<int> mov_list;
    std::unordered_map<int, std::pair<int, int>> instance_category_area_map;
    std::unordered_map<int, std::pair<int, int>> semantics_category_area_map;

    // Point cloud segments
    std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr> clouds;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

    // Publishers
    ros::Publisher point_cloud_segment_publisher_;
    ros::Publisher pano_seg_image_publisher_;

    // Node handle
    ros::NodeHandle node_handle_;

    // segmenters
    PCProcessor* mpc_processor;
    depth_segmentation::DepthSegmenter* mdepth_segmenter;

    bool verbose;


};


}


#endif