
#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

/* This example shows how to estimate the SIFT points based on the
 * Normal gradients i.e. curvature than using the Intensity gradient
 * as usually used for SIFT keypoint estimation.
 */

int
main(int, char** argv)
{
  std::string filename = argv[1];
  std::cout << "Reading " << filename << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  // if(pcl::io::loadPLYFile<pcl::PointXYZ> (filename, *cloud_xyz) == -1) // load the file
  if(pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud_xyz) == -1) // load the file
  {
    PCL_ERROR("Couldn't read file\n");
    return -1;
  }
  std::cout << "points: " << cloud_xyz->size () <<std::endl;
  
  // Parameters for sift computation
  const float min_scale = 0.01f;
  const int n_octaves = 6;
  const int n_scales_per_octave = 4;
  const float min_contrast = 0.005f;
  
  // Estimate the normals of the cloud_xyz
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());

  ne.setInputCloud(cloud_xyz);
  ne.setSearchMethod(tree_n);
  ne.setRadiusSearch(10);
  ne.compute(*cloud_normals);

  // Copy the xyz info from cloud_xyz and add it to cloud_normals as the xyz field in PointNormals estimation is zero
  for(std::size_t i = 0; i<cloud_normals->size(); ++i)
  {
    (*cloud_normals)[i].x = (*cloud_xyz)[i].x;
    (*cloud_normals)[i].y = (*cloud_xyz)[i].y;
    (*cloud_normals)[i].z = (*cloud_xyz)[i].z;
  }

  // Estimate the sift interest points using normals values from xyz as the Intensity variants
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud_normals);
  sift.compute(result);

  std::cout << "No of SIFT points in the result are " << result.size () << std::endl;


  // Copying the pointwithscale to pointxyz so as visualize the cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *cloud_temp);
  std::cout << "SIFT points in the cloud_temp are " << cloud_temp->size () << std::endl;


  // Visualization of keypoints along with the original cloud
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (cloud_temp, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler (cloud_xyz, 255, 0, 0);
  viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
  viewer.addPointCloud(cloud_xyz, cloud_color_handler, "cloud");
  viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

  while(!viewer.wasStopped ())
  {
  viewer.spinOnce ();
  }



  return 0;
  
}
