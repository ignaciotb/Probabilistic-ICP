#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/gicp.h>

#include <pcl/registration/warp_point_rigid.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/conversions.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/transformation_estimation_svd.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace pcl::registration;
PointCloud<PointXYZ>::Ptr src, tgt;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
using namespace std;

/* This examples shows how to estimate the SIFT points based on the 
 * z gradient of the 3D points than using the Intensity gradient as
 * usually used for SIFT keypoint estimation.
 */

int v1 (0);
int v2 (1);

namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

bool next_iteration_icp = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing){
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration_icp = true;
}

void plotCorrespondences(pcl::visualization::PCLVisualizer& viewer, 
                         pcl::Correspondences& corrs, 
                         PointCloudT::Ptr& src, 
                         PointCloudT::Ptr& trg){

    // Plot initial trajectory estimate
    // viewer.removeAllCoordinateSystems(v1);
    // viewer.addCoordinateSystem(5.0, "reference_frame", v1);
    viewer.addCoordinateSystem(5.0, src->sensor_origin_(0), src->sensor_origin_(1), 
                                src->sensor_origin_(2), "submap_1", v1);

    viewer.addCoordinateSystem(5.0, trg->sensor_origin_(0), trg->sensor_origin_(1), 
                            trg->sensor_origin_(2), "submap_2", v1);

    int j = 0;
    Eigen::Vector3i dr_color = Eigen::Vector3i(rand() % 256, rand() % 256, rand() % 256);
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf (0, 3) = src->sensor_origin_(0);
    tf (1, 3) = src->sensor_origin_(1);
    tf (2, 3) = src->sensor_origin_(2);
    for(auto corr_i: corrs){
        Eigen::Vector4f p_src = tf * src->at(corr_i.index_query).getVector4fMap();
        Eigen::Vector4f p_trg = tf * trg->at(corr_i.index_match).getVector4fMap();
        viewer.addLine(PointT(p_src(0), p_src(1), p_src(2)), PointT(p_trg(0), p_trg(1), p_trg(2)),
                dr_color[0], dr_color[1], dr_color[2], "corr_" + std::to_string(j), v1);
        j++;
    }
    viewer.spinOnce();
}

////////////////////////////////////////////////////////////////////////////////
void
estimateKeypoints (const PointCloud<PointXYZ>::Ptr &src, 
                   const PointCloud<PointXYZ>::Ptr &tgt,
                   PointCloud<PointXYZ> &keypoints_src,
                   PointCloud<PointXYZ> &keypoints_tgt)
{
  // Get an uniform grid of keypoints
  UniformSampling<PointXYZ> uniform;
  uniform.setRadiusSearch (1);  // 1m

  uniform.setInputCloud (src);
  uniform.filter (keypoints_src);

  uniform.setInputCloud (tgt);
  uniform.filter (keypoints_tgt);

  // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
  // pcl_viewer source_pcd keypoints_src.pcd -ps 1 -ps 10
  savePCDFileBinary ("keypoints_src.pcd", keypoints_src);
  savePCDFileBinary ("keypoints_tgt.pcd", keypoints_tgt);
}

////////////////////////////////////////////////////////////////////////////////
void
estimateNormals (const PointCloud<PointXYZ>::Ptr &src, 
                 const PointCloud<PointXYZ>::Ptr &tgt,
                 PointCloud<Normal> &normals_src,
                 PointCloud<Normal> &normals_tgt)
{
  NormalEstimation<PointXYZ, Normal> normal_est;
  normal_est.setInputCloud (src);
  normal_est.setRadiusSearch (0.5);  // 50cm
  normal_est.compute (normals_src);

  normal_est.setInputCloud (tgt);
  normal_est.compute (normals_tgt);

  // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
  // pcl_viewer normals_src.pcd
  PointCloud<PointNormal> s, t;
  copyPointCloud (*src, s);
  copyPointCloud (normals_src, s);
  copyPointCloud (*tgt, t);
  copyPointCloud (normals_tgt, t);
  savePCDFileBinary ("normals_src.pcd", s);
  savePCDFileBinary ("normals_tgt.pcd", t);
}

////////////////////////////////////////////////////////////////////////////////
void
estimateFPFH (const PointCloud<PointXYZ>::Ptr &src, 
              const PointCloud<PointXYZ>::Ptr &tgt,
              const PointCloud<Normal>::Ptr &normals_src,
              const PointCloud<Normal>::Ptr &normals_tgt,
              const PointCloud<PointXYZ>::Ptr &keypoints_src,
              const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
              PointCloud<FPFHSignature33> &fpfhs_src,
              PointCloud<FPFHSignature33> &fpfhs_tgt)
{
  FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
  fpfh_est.setInputCloud (keypoints_src);
  fpfh_est.setInputNormals (normals_src);
  fpfh_est.setRadiusSearch (1); // 1m
  fpfh_est.setSearchSurface (src);
  fpfh_est.compute (fpfhs_src);

  fpfh_est.setInputCloud (keypoints_tgt);
  fpfh_est.setInputNormals (normals_tgt);
  fpfh_est.setSearchSurface (tgt);
  fpfh_est.compute (fpfhs_tgt);

  // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
  // pcl_viewer fpfhs_src.pcd
  PCLPointCloud2 s, t, out;
  toPCLPointCloud2 (*keypoints_src, s); toPCLPointCloud2 (fpfhs_src, t); concatenateFields (s, t, out);
  savePCDFile ("fpfhs_src.pcd", out);
  toPCLPointCloud2 (*keypoints_tgt, s); toPCLPointCloud2 (fpfhs_tgt, t); concatenateFields (s, t, out);
  savePCDFile ("fpfhs_tgt.pcd", out);
}

////////////////////////////////////////////////////////////////////////////////
void
findCorrespondences (const PointCloud<FPFHSignature33>::Ptr &fpfhs_src,
                     const PointCloud<FPFHSignature33>::Ptr &fpfhs_tgt,
                     Correspondences &all_correspondences)
{
  CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> est;
  est.setInputCloud (fpfhs_src);
  est.setInputTarget (fpfhs_tgt);
  est.determineReciprocalCorrespondences (all_correspondences);
}

////////////////////////////////////////////////////////////////////////////////
void
rejectBadCorrespondences (const CorrespondencesPtr &all_correspondences,
                          const PointCloud<PointXYZ>::Ptr &keypoints_src,
                          const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
                          Correspondences &remaining_correspondences)
{
  CorrespondenceRejectorDistance rej;
  rej.setInputSource<PointXYZ> (keypoints_src);
  rej.setInputTarget<PointXYZ> (keypoints_tgt);
  rej.setMaximumDistance (5);    // 1m
  rej.setInputCorrespondences (all_correspondences);
  rej.getCorrespondences (remaining_correspondences);
}


////////////////////////////////////////////////////////////////////////////////
void
computeTransformation (const PointCloud<PointXYZ>::Ptr &src, 
                       const PointCloud<PointXYZ>::Ptr &tgt,
                       Eigen::Matrix4f &transform)
{
  // Get an uniform grid of keypoints
  PointCloud<PointXYZ>::Ptr keypoints_src (new PointCloud<PointXYZ>), 
                            keypoints_tgt (new PointCloud<PointXYZ>);

  estimateKeypoints (src, tgt, *keypoints_src, *keypoints_tgt);
  print_info ("Found %zu and %zu keypoints for the source and target datasets.\n", static_cast<std::size_t>(keypoints_src->size ()), static_cast<std::size_t>(keypoints_tgt->size ()));

  // Compute normals for all points keypoint
  PointCloud<Normal>::Ptr normals_src (new PointCloud<Normal>), 
                          normals_tgt (new PointCloud<Normal>);
  estimateNormals (src, tgt, *normals_src, *normals_tgt);
  print_info ("Estimated %zu and %zu normals for the source and target datasets.\n", static_cast<std::size_t>(normals_src->size ()), static_cast<std::size_t>(normals_tgt->size ()));

  // Compute FPFH features at each keypoint
  PointCloud<FPFHSignature33>::Ptr fpfhs_src (new PointCloud<FPFHSignature33>), 
                                   fpfhs_tgt (new PointCloud<FPFHSignature33>);
  estimateFPFH (src, tgt, normals_src, normals_tgt, keypoints_src, keypoints_tgt, *fpfhs_src, *fpfhs_tgt);

  // Copy the data and save it to disk
/*  PointCloud<PointNormal> s, t;
  copyPointCloud (*keypoints_src, s);
  copyPointCloud (normals_src, s);
  copyPointCloud (*keypoints_tgt, t);
  copyPointCloud (normals_tgt, t);*/

  // Find correspondences between keypoints in FPFH space
  CorrespondencesPtr all_correspondences (new Correspondences), 
                     good_correspondences (new Correspondences);
  findCorrespondences (fpfhs_src, fpfhs_tgt, *all_correspondences);

  // Reject correspondences based on their XYZ distance
  rejectBadCorrespondences (all_correspondences, keypoints_src, keypoints_tgt, *good_correspondences);

  std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
  std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
  for (const auto& corr : (*good_correspondences))
    std::cerr << corr << std::endl;
  // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
  TransformationEstimationSVD<PointXYZ, PointXYZ> trans_est;
  trans_est.estimateRigidTransformation (*keypoints_src, *keypoints_tgt, *all_correspondences, transform);
}

void pclVisualizer(pcl::visualization::PCLVisualizer& viewer,
                   const PointCloudT::Ptr cloud_in,
                   const PointCloudT::Ptr cloud_tr,
                   const PointCloudT::Ptr cloud_icp){

    // Viewports
    viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (cloud_in, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                               (int) 255 * txt_gray_lvl);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (cloud_tr, 20, 180, 20);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (cloud_icp, 180, 20, 20);
    
    viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_src_v1", v1);
    viewer.addPointCloud (cloud_tr, cloud_tr_color_h, "cloud_trg_v1", v1);
    viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_src_v2", v2);
    viewer.addPointCloud (cloud_icp, cloud_icp_color_h, "cloud_trg_v2", v2);

    // Text descriptions and background
    viewer.addText ("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    viewer.addText ("White: Original point cloud\nRed: PICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);
    viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    // Set camera position and orientation
    viewer.setCameraPosition (-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize (1280, 1024);  // Visualiser window size
    viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);

}

void runGicp(PointCloudT::Ptr& src_cloud, const PointCloudT::Ptr& trg_cloud){
    
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    
    // Constrain GICP to x,y, yaw
    pcl::registration::WarpPointRigid3D<PointT, PointT>::Ptr warp_fcn
      (new pcl::registration::WarpPointRigid3D<PointT, PointT>);

    pcl::registration::TransformationEstimationLM<PointT, PointT>::Ptr te
            (new pcl::registration::TransformationEstimationLM<PointT, PointT>);
    te->setWarpFunction (warp_fcn);
    gicp.setTransformationEstimation(te);
    
    gicp.setInputSource(src_cloud);
    gicp.setInputTarget(trg_cloud);

    gicp.setMaxCorrespondenceDistance(40);
    gicp.setMaximumIterations(200);
    // gicp.setMaximumOptimizerIterations(200);
    // gicp.setRANSACIterations(100);
    // gicp.setRANSACOutlierRejectionThreshold(10);
    gicp.setTransformationEpsilon(1e-4);
    // gicp.setUseReciprocalCorrespondences(true);

    gicp.align(*src_cloud);
}

void computeSiftFeatures(const PointCloudT::Ptr cloud_in, pcl::PointCloud<pcl::PointWithScale>& result){

    // Parameters for sift computation
    const float min_scale = 1.f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.005f;

    // Estimate the sift interest points using z values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_in);
    sift.compute(result);

    std::cout << "No of SIFT points in the result are " << result.size() << std::endl;
}

int main(int, char **argv)
{
    // Parse the command line arguments for .pcd files
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trg(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);

    // Load the files
    if (pcl::io::loadPCDFile (argv[1], *cloud_src) < 0){
        PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }

    if (pcl::io::loadPCDFile (argv[2], *cloud_trg) < 0){
        PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }

    // Initial misalignment between pointclouds
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    double theta = M_PI / 5;
    transformation_matrix (0, 0) = cos (theta);
    transformation_matrix (0, 1) = -sin (theta);
    transformation_matrix (1, 0) = sin (theta);
    transformation_matrix (1, 1) = cos (theta);
//    transformation_matrix (0, 3) = -0.2;
//    transformation_matrix (1, 3) = -0.4;
//    transformation_matrix (2, 3) = -0.2;
    pcl::transformPointCloud(*cloud_src, *cloud_icp, transformation_matrix);
    // *cloud_trg = *cloud_trg;

    // Extract SIFT features of both submaps
    pcl::PointCloud<pcl::PointWithScale> result_src, result_trg;
    computeSiftFeatures(cloud_src, result_src);
    computeSiftFeatures(cloud_trg, result_trg);

    // Initialize viewer object (use same while loop as in ICP PCL example?)
    pcl::visualization::PCLVisualizer viewer ("Probabilistic ICP demo");
    pclVisualizer(viewer, cloud_src, cloud_trg, cloud_icp);
    
    // Visualize SIFT point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp_src (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp_trg (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints1_color_handler (cloud_temp_src, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints2_color_handler (cloud_temp_trg, 255, 0, 0);

    // Copying the pointwithscale to pointxyz so as visualize the cloud
    copyPointCloud(result_src, *cloud_temp_src);
    std::cout << "SIFT points in the result are " << cloud_temp_src->size () << std::endl;
    viewer.addPointCloud(cloud_temp_src, keypoints1_color_handler, "keypoints_src", v1);
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_src");
    
    copyPointCloud(result_trg, *cloud_temp_trg);
    std::cout << "SIFT points in the result are " << cloud_temp_trg->size () << std::endl;
    viewer.addPointCloud(cloud_temp_trg, keypoints2_color_handler, "keypoints_trg", v1);
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_trg");

    // Update visualizer
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (cloud_src, 180, 20, 20);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (cloud_src, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                               (int) 255 * txt_gray_lvl);

    // // Initial misalignment between pointclouds
    // Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    // double theta = M_PI / 5;
    // // transformation_matrix (0, 0) = cos (theta);
    // // transformation_matrix (0, 1) = -sin (theta);
    // // transformation_matrix (1, 0) = sin (theta);
    // // transformation_matrix (1, 1) = cos (theta);
    // transformation_matrix (0, 3) = -0.2;
    // transformation_matrix (1, 3) = -0.4;
    // transformation_matrix (2, 3) = 0.;
    // pcl::transformPointCloud(*cloud_src, *cloud_src, transformation_matrix);
    // viewer.updatePointCloud(cloud_src, cloud_in_color_h, "cloud_src_v2");

    // Basic correspondence estimation between SIFT features
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
    pcl::Correspondences all_correspondences;
    all_correspondences.clear();
    est.setInputSource(cloud_src);
    est.setInputTarget(cloud_trg); 
    est.determineReciprocalCorrespondences(all_correspondences, 20.0);
    plotCorrespondences(viewer, all_correspondences, cloud_src, cloud_trg);

    while(!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
    viewer.resetStoppedFlag();

    // Run GICP
    // std::cout << "RUnning gicp" << std::endl;
    // runGicp(cloud_src, cloud_trg);

    // viewer.updatePointCloud(cloud_src, cloud_in_color_h, "cloud_src_v2");
   

    // // Compute the best transformtion
    // Eigen::Matrix4f transform;
    // computeTransformation (src, tgt, transform);

    // std::cerr << transform << std::endl;
    // // Transform the data and write it to disk
    // PointCloud<PointXYZ> output;
    // transformPointCloud (*src, output, transform);
    // PointCloudT::Ptr output_ptr;
    // output_ptr.reset(new PointCloudT(output));

    // viewer.updatePointCloud(output_ptr, cloud_icp_color_h, "cloud_icp_v2");

    // while(!viewer.wasStopped ())
    // {
    //     viewer.spinOnce ();
    // }

    return 0;
}

