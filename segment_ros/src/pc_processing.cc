#include "common.h"
#include "pc_processing.h"
// #include "utils.h"
namespace SegmentROS
{

PCProcessor::PCProcessor(ros::NodeHandle& node_handle, bool visualize_geo_seg): if_visualize(visualize_geo_seg)
{
    // load params
    node_handle.param<float>("lccp/voxel_resolution", lccp_param_.voxel_resolution, 0.008f);
    node_handle.param<float>("lccp/seed_resolution", lccp_param_.seed_resolution, 0.05f);
    node_handle.param<float>("lccp/color_importance", lccp_param_.color_importance, 0.0f);
    node_handle.param<float>("lccp/spatial_importance", lccp_param_.spatial_importance, 1.0f);
    node_handle.param<float>("lccp/normal_importance", lccp_param_.normal_importance, 4.0f);
    node_handle.param<bool>("lccp/use_supervoxel_refinement", lccp_param_.use_supervoxel_refinement, false);

    node_handle.param<float>("lccp/concavity_tolerance_threshold", lccp_param_.concavity_tolerance_threshold, 10);
    node_handle.param<float>("lccp/smoothness_threshold", lccp_param_.smoothness_threshold, 0.1);
    node_handle.param<int>("lccp/min_segment_size", lccp_param_.min_segment_size, 3);
    node_handle.param<bool>("lccp/use_extended_convexity", lccp_param_.use_extended_convexity, false);
    node_handle.param<bool>("lccp/use_sanity_criterion", lccp_param_.use_sanity_criterion, false);

    node_handle.param<bool>("lccp/show_supervoxels", lccp_param_.show_supervoxels, false);

    if (!lccp_param_.use_extended_convexity)
        lccp_param_.k_factor = 0;
    else
        lccp_param_.k_factor = 1;

    node_handle.param<float>("voxel_filter/leaf_size", filter_param_.leaf_size, 0.005f);
    node_handle.param<bool>("voxel_filter/use_voxel_downsample", filter_param_.use_voxel_downsample, true);

    node_handle.param<float>("Euclidean_seg/cluster_tolerance", filter_param_.cluster_tolerance, 0.05f);
    node_handle.param<int>("Euclidean_seg/min_cluster_size", filter_param_.min_cluster_size, 10);
    node_handle.param<int>("Euclidean_seg/max_cluster_size", filter_param_.max_cluster_size, 25000);
    node_handle.param<float>("normal/search_radius", filter_param_.search_radius, 0.03f);

    ROS_INFO("Initialize pc_processor!");

    // initialize filters and segmentors
    tree_label.reset (new pcl::search::KdTree<PointSurfelLabel>);
    tree_normal.reset (new pcl::search::KdTree<pcl::PointXYZRGB>);
        
}


void PCProcessor::SetCamIntrinsics(const cv::Mat& K)
{
    K.copyTo(K_);
}
  

void PCProcessor::GeneratePC(const cv::Mat& imRGB, const cv::Mat& imDepth, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
    if (imRGB.empty() || imDepth.empty()){
        std::cout << "No image available" << std::endl;
        return;
    }

    if (imRGB.rows!=imDepth.rows || imRGB.cols!=imDepth.cols){
        std::cout << "Image size doesn't match" << std::endl;
        return;
    }

    if (K_.empty()){
        std::cout << "Camera intrinsics has not been set" <<std::endl;
        return;
    }

    int height = imDepth.rows;
    int width = imDepth.cols;
    register int left, right;
    int top, bottom;

    cloud->height = height;
    cloud->width = width;

    cloud->is_dense = false;
    cloud->points.resize (cloud->height * cloud->width);
    left = 0;
    right = width;
    top = 0;
    bottom = height;
    
    register float inv_fx = 1.0f / K_.at<float>(0,0);
    register float inv_fy = 1.0f / K_.at<float>(1,1);

    register int centerX = K_.at<float>(0,2);
    register int centerY = K_.at<float>(1,2);

    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    
    register int depth_idx = 0;
    for (int v = top; v < bottom; ++v)
    {
        for (register int u = left; u < right; ++u, ++depth_idx)
        {
            pcl::PointXYZRGB& pt = cloud->points[depth_idx];
        
            if (imDepth.at<float>(v,u) == 0.0)
            {
                pt.x = pt.y = pt.z = bad_point;
                continue;
            }
        
            pt.z = imDepth.at<float>(v,u);
            pt.x = static_cast<float> (u - centerX) * pt.z * inv_fx;
            pt.y = static_cast<float> (v - centerY) * pt.z * inv_fy;

            pt.r = imRGB.at<cv::Vec3b>(v,u)[0];
            pt.g = imRGB.at<cv::Vec3b>(v,u)[1];
            pt.b = imRGB.at<cv::Vec3b>(v,u)[2];
        }
    }

    cloud->sensor_origin_.setZero ();
    cloud->sensor_orientation_.w () = 0.0f;
    cloud->sensor_orientation_.x () = 1.0f;
    cloud->sensor_orientation_.y () = 0.0f;
    cloud->sensor_orientation_.z () = 0.0f;  
}


void PCProcessor::GeneratePCSem (const cv::Mat& imRGB, const cv::Mat& imDepth, const cv::Mat& imSeg, const std::vector<Obj2D>& objects, 
                                      const std::vector<Sem2D>& semantics, const std::vector<int>& mov_list,
                                      std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds)
{
    if (imRGB.empty() || imDepth.empty() || imSeg.empty()){
        std::cout << "No image available" << std::endl;
        return;
    }

    if (imRGB.rows!=imDepth.rows || imRGB.cols!=imDepth.cols){
        std::cout << "Image size doesn't match" << std::endl;
        return;
    }

    if (K_.empty()){
        std::cout << "Camera intrinsics has not been set" <<std::endl;
        return;
    }

    int height = imDepth.rows;
    int width = imDepth.cols;

    cv::Mat mask = cv::Mat::ones(height, width, CV_8UC1);

    register int left, right;
    register int top, bottom;

    // Get static semantics/instances
    std::vector<int> seg_id = {};
    std::vector<int> cate_id = {};
    std::vector<cv::Rect> seg_bb = {};
    std::vector<int> mov_id = {};
    std::vector<cv::Rect> mov_bb = {};
    for (int i = 0; i < objects.size(); i++)
    {
        if (!objects[i].is_dynamic)
        {
            seg_id.push_back(objects[i].id);
            cate_id.push_back(objects[i].category);
            seg_bb.push_back(objects[i].box);
        }
        else
        {
            mov_id.push_back(objects[i].id);
            mov_bb.push_back(objects[i].box);
        }
    }

    for (int i = 0; i < semantics.size(); i++)
    {
        seg_id.push_back(semantics[i].id);
        cate_id.push_back(semantics[i].category);
    }

    // Initialize point cloud segments
    clouds.reserve(seg_id.size()+1);
    for (int i = 0; i < seg_id.size()+1; i++)
    {
        pcl::PointCloud<PointSurfelLabel>::Ptr cloud(new pcl::PointCloud<PointSurfelLabel>);
        
        if (i < seg_bb.size())
        {
            cloud->height = seg_bb[i].height;
            cloud->width = seg_bb[i].width;
            cloud->points.resize (cloud->height * cloud->width);
        }
        else
        {
            cloud->height = height;
            cloud->width = width;
            cloud->points.resize (cloud->height * cloud->width);
        } 
        
        cloud->is_dense = false;
        cloud->sensor_origin_.setZero ();
        cloud->sensor_orientation_.w () = 0.0f;
        cloud->sensor_orientation_.x () = 1.0f;
        cloud->sensor_orientation_.y () = 0.0f;
        cloud->sensor_orientation_.z () = 0.0f;  

        clouds.push_back(cloud);
    }

    // Fill in point clouds
    register float inv_fx = 1.0f / K_.at<float>(0,0);
    register float inv_fy = 1.0f / K_.at<float>(1,1);

    register int centerX = K_.at<float>(0,2);
    register int centerY = K_.at<float>(1,2);

    float bad_point = std::numeric_limits<float>::quiet_NaN ();

    register int depth_idx = 0;

    for (int k = 0; k < seg_id.size() + 1; k++)
    {
        depth_idx = 0;

        left = 0;
        right = width;
        top = 0;
        bottom = height;
        
        if (k < seg_id.size())
        {
            if (k < seg_bb.size())
            {
                left = seg_bb[k].x;
                right = left + seg_bb[k].width;
                top = seg_bb[k].y;
                bottom = top + seg_bb[k].height;
            }
            
            for (register int v = top; v < bottom; ++v)
            {
                for (register int u = left; u < right; ++u, ++depth_idx)
                {
                    int label = (int)(imSeg.at<uchar>(v,u));
                    PointSurfelLabel& pt = clouds[k]->points[depth_idx];
                    // if (mask.at<uchar>(v,u) == 0)
                    //   pt.x = pt.y = pt.z = bad_point; 
                    if (imDepth.at<float>(v,u) == 0.0 || !(imDepth.at<float>(v,u)))
                    {
                        pt.x = pt.y = pt.z = bad_point; 
                        mask.at<uchar>(v,u) = 0; 
                    }
                    else if(label == seg_id[k])
                    {
                        pt.z = imDepth.at<float>(v,u);
                        pt.x = static_cast<float> (u - centerX) * pt.z * inv_fx;
                        pt.y = static_cast<float> (v - centerY) * pt.z * inv_fy;

                        pt.r = imRGB.at<cv::Vec3b>(v,u)[0];
                        pt.g = imRGB.at<cv::Vec3b>(v,u)[1];
                        pt.b = imRGB.at<cv::Vec3b>(v,u)[2];
                        pt.a = 255;

                        pt.instance_label = (uint8_t)(label);
                        pt.semantic_label = (uint8_t)(cate_id[k]);

                        mask.at<uchar>(v,u) = 0; 
                    }
                    else
                        pt.x = pt.y = pt.z = bad_point; 

                }
            }
        }

        else  //k=seg_id.size() background
        {
            for (register int v = top; v < bottom; ++v)
            {
                for (register int u = left; u < right; ++u, ++depth_idx)
                {
                    int label = (int)(imSeg.at<uchar>(v,u));
                    PointSurfelLabel& pt = clouds[k]->points[depth_idx];
                    if ((imDepth.at<float>(v,u) == 0.0) || !(imDepth.at<float>(v,u)) || (mask.at<uchar>(v,u) == 0))
                        pt.x = pt.y = pt.z = bad_point;  
                    else if (IsInBB(u ,v, mov_bb))
                        pt.x = pt.y = pt.z = bad_point;  
                    else
                    {
                        // pt.x = pt.y = pt.z = bad_point;  
                        pt.z = imDepth.at<float>(v,u);
                        pt.x = static_cast<float> (u - centerX) * pt.z * inv_fx;
                        pt.y = static_cast<float> (v - centerY) * pt.z * inv_fy;

                        pt.r = imRGB.at<cv::Vec3b>(v,u)[0];
                        pt.g = imRGB.at<cv::Vec3b>(v,u)[1];
                        pt.b = imRGB.at<cv::Vec3b>(v,u)[2];
                        pt.a = 255;
                        pt.instance_label = 0u;
                        pt.semantic_label = background_label;
                    }
                }
            }
        }

        std::vector<int> map;
        pcl::removeNaNFromPointCloud(*(clouds[k]),*(clouds[k]),map);

    }
}


void PCProcessor::FilterPCVoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filtered)
{
    if (!cloud->is_dense)
    {
        std::vector<int> map;
        pcl::removeNaNFromPointCloud(*(cloud),*(cloud),map);
        cloud->is_dense = true;
    }

    if (cloud->points.size() > 0)
    {   
        // Through voxel grid filtering, the field of instance/semantic labels are lost
        vg.setLeafSize (filter_param_.leaf_size, filter_param_.leaf_size, filter_param_.leaf_size);
        vg.setInputCloud (cloud);
        vg.filter (*cloud_filtered);
    }
}


void PCProcessor::FilterPCEuclidean(pcl::PointCloud<PointSurfelLabel>::Ptr& cloud,
                                   pcl::PointCloud<PointSurfelLabel>::Ptr& cloud_filtered)
{
    
    if (!cloud->is_dense)
    {
        std::vector<int> map;
        pcl::removeNaNFromPointCloud(*(cloud),*(cloud),map);
        cloud->is_dense = true;
    }

    if (cloud->points.size() > 0)
    {
        tree_label->setInputCloud (cloud);
        std::vector<pcl::PointIndices> cluster_indices;

        ec.setClusterTolerance (filter_param_.cluster_tolerance);
        ec.setMinClusterSize (filter_param_.min_cluster_size);
        ec.setMaxClusterSize (filter_param_.max_cluster_size);
        ec.setSearchMethod (tree_label);
        ec.setInputCloud (cloud_filtered);
        ec.extract (cluster_indices);

        // std::cout<<cluster_indices.size()<<std::endl;

        if (cluster_indices.size() > 0)
        {
            std::vector<int> cluster_size;
            for (int j = 0; j < cluster_indices.size(); j++)
            cluster_size.push_back(cluster_indices[j].indices.size());
            
            auto it = max_element(std::begin(cluster_size), std::end(cluster_size));
            pcl::PointIndices instance_indices = cluster_indices[it-cluster_size.begin()];

            pcl::PointCloud<PointSurfelLabel>::Ptr cloud_cluster (new pcl::PointCloud<PointSurfelLabel>);
            for (std::vector<int>::const_iterator pit = instance_indices.indices.begin (); pit != instance_indices.indices.end (); ++pit)
            {
                cloud_filtered->points[*pit].instance_label = uint8_t(cloud->points[0].instance_label);
                cloud_filtered->points[*pit].semantic_label = uint8_t(cloud->points[0].semantic_label);

                cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
            }
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            *cloud_filtered = *cloud_cluster;      
            
        }
        else
        {
            *cloud_filtered = *cloud;      
            std::cout << "Fail to perform Euclidean Clustering" << std::endl;
        }
    }
}


void PCProcessor::ComputeNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& normal_cloud)
{

    if (!cloud->is_dense)
    {
        std::vector<int> map;
        pcl::removeNaNFromPointCloud(*(cloud),*(cloud),map);
        cloud->is_dense = true;
    }

    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    ne.setSearchMethod (tree_normal);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 
    ne.setRadiusSearch (filter_param_.search_radius);

    // Compute the features
    ne.compute (*cloud_normals);
    pcl::concatenateFields (*cloud, *cloud_normals, *normal_cloud);

}


void PCProcessor::SegmentPCGeo(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds)                             
{

    std::string output_folder1 = "/home/roma/dev/catkin_perception_ros/src/perception_ros/perception_ros/output/test/pcd/";
    if (!cloud->is_dense)
    {
        std::vector<int> map;
        pcl::removeNaNFromPointCloud(*(cloud),*(cloud),map);
        cloud->is_dense = true;
    }

    using PointT = pcl::PointXYZRGB;  // The point type used for input
    using SuperVoxelAdjacencyList = pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList;
    using VertexIterator = pcl::LCCPSegmentation<PointT>::VertexIterator;

    float normals_scale = lccp_param_.seed_resolution / 2.0;

    /// Preparation of Input: Supervoxel Oversegmentation

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud(*cloud, *cloud_normal);

    pcl::PointCloud<PointT>::Ptr cloud_input(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, *cloud_input);

    pcl::SupervoxelClustering<pcl::PointXYZRGB> super (lccp_param_.voxel_resolution, lccp_param_.seed_resolution);

    super.setUseSingleCameraTransform (lccp_param_.use_single_cam_transform);
    super.setInputCloud (cloud_input);
    super.setNormalCloud (cloud_normal);

    super.setColorImportance (lccp_param_.color_importance);
    super.setSpatialImportance (lccp_param_.spatial_importance);
    super.setNormalImportance (lccp_param_.normal_importance);
    std::map<std::uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

    // PCL_INFO ("Extracting supervoxels\n");
    super.extract (supervoxel_clusters);

    if (lccp_param_.use_supervoxel_refinement)
    {
        // PCL_INFO ("Refining supervoxels\n");
        super.refineSupervoxels (2, supervoxel_clusters);
    }
    // std::stringstream temp;
    // temp << "  Nr. Supervoxels: " << supervoxel_clusters.size () << "\n";
    // PCL_INFO (temp.str ().c_str ());

    // PCL_INFO ("Getting supervoxel adjacency\n");
    std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);

    /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
    pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);

    /// The Main Step: Perform LCCPSegmentation

    // PCL_INFO ("Starting Segmentation\n");
    lccp.setConcavityToleranceThreshold (lccp_param_.concavity_tolerance_threshold);
    lccp.setSanityCheck (lccp_param_.use_sanity_criterion);
    lccp.setSmoothnessCheck (true, lccp_param_.voxel_resolution, lccp_param_.seed_resolution, lccp_param_.smoothness_threshold);
    lccp.setKFactor (lccp_param_.k_factor);
    lccp.setInputSupervoxels (supervoxel_clusters, supervoxel_adjacency);
    lccp.setMinSegmentSize (lccp_param_.min_segment_size);
    lccp.segment ();

    // PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");
    pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud ();
    pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = sv_labeled_cloud->makeShared ();
    lccp.relabelCloud (*lccp_labeled_cloud);
    SuperVoxelAdjacencyList sv_adjacency_list;
    lccp.getSVAdjacencyList (sv_adjacency_list);  // Needed for visualization

    std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
    vertex_iterator_range = boost::vertices (sv_adjacency_list);

    /// Creating clouds
    if (lccp_labeled_cloud->size () == cloud->size ())
    {
        std::vector<int> labels = {};
        clouds.clear();

        for (int i = 0; i < lccp_labeled_cloud->points.size(); i++) 
        {
            int label = (int)lccp_labeled_cloud->points[i].label;
            auto it = std::find(labels.begin(), labels.end(), label);
            PointSurfelLabel pt;
            pcl::copyPoint(cloud->points[i], pt);
        
            if (it == labels.end())
            {
                labels.push_back(label);
                pcl::PointCloud<PointSurfelLabel>::Ptr cloud_temp(new pcl::PointCloud<PointSurfelLabel>);
            
                cloud_temp->is_dense = false;
                cloud_temp->sensor_origin_.setZero ();
                cloud_temp->sensor_orientation_.w () = 0.0f;
                cloud_temp->sensor_orientation_.x () = 1.0f;
                cloud_temp->sensor_orientation_.y () = 0.0f;
                cloud_temp->sensor_orientation_.z () = 0.0f;  

                cloud_temp->push_back(pt);
                clouds.push_back(cloud_temp);
            }
            else
                clouds[it-labels.begin()]->push_back(pt);
        }

        if (if_visualize)
        {
            using namespace pcl;

            using AdjacencyIterator = LCCPSegmentation<PointT>::AdjacencyIterator;
            using EdgeID = LCCPSegmentation<PointT>::EdgeID;

            if (if_first_time)
            {
                viewer.reset(new pcl::visualization::PCLVisualizer ("Point Cloud Segmentation Viewer"));
                viewer->setBackgroundColor (0, 0, 0);
                // viewer->addCoordinateSystem (1.0);
                viewer->initCameraParameters ();
                viewer->addPointCloud (lccp_labeled_cloud, "maincloud");
                if_first_time = false;
            }
            else
            /// Show Segmentation or Supervoxels
                viewer->updatePointCloud ( (lccp_param_.show_supervoxels) ? sv_labeled_cloud : lccp_labeled_cloud, "maincloud");

            viewer->spinOnce(10); // Will crash if use multi-thread spinning in ros 
            // makePath(output_folder + "/pcd/", 0777);
            // pcl::io::savePCDFile("/home/roma/dev/catkin_perception_ros/src/perception_ros/perception_ros/output/test.pcd", *lccp_labeled_cloud);
            // std::cout<<output_folder1+std::to_string(NFrame)+".pcd"<<endl;
            pcl::io::savePCDFile(output_folder1+std::to_string(NFrame)+".pcd", *lccp_labeled_cloud);
            // std::cout<<"========save===pcd===file"<<endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }  
    }        
    else
    {
        PCL_ERROR ("ERROR:: Sizes of input cloud and labeled supervoxel cloud do not match. No output is produced.\n");
    }

}


void PCProcessor::SegmentSingleFrame (const cv::Mat& imRGB, const cv::Mat& imDepth, std::vector<pcl::PointCloud<PointSurfelLabel>::Ptr>& clouds)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    GeneratePC(imRGB, imDepth, cloud);

    if (filter_param_.use_voxel_downsample)
        FilterPCVoxelGrid(cloud, cloud);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ComputeNormal(cloud, normal_cloud);

    SegmentPCGeo(normal_cloud, clouds); 
}

}
