#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

#include <eigen3/Eigen/Core>

#include "p_icp/utils.hpp"
#include "p_icp/picp_solver.hpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace std;

bool next_iteration_icp = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing){
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration_icp = true;
}

void pclVisualizer(pcl::visualization::PCLVisualizer& viewer,
                   const PointCloudT::Ptr cloud_in,
                   const PointCloudT::Ptr cloud_tr,
                   const PointCloudT::Ptr cloud_icp){

    // Viewports
    int v1 (0);
    int v2 (1);
    viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (cloud_in, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                               (int) 255 * txt_gray_lvl);
    viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_in_v2", v2);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (cloud_tr, 20, 180, 20);
    viewer.addPointCloud (cloud_tr, cloud_tr_color_h, "cloud_tr_v1", v1);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (cloud_icp, 180, 20, 20);
    viewer.addPointCloud (cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

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

int main (int argc, char* argv[]) {
    // The point clouds we will be using
    PointCloudT::Ptr cloud_in (new PointCloudT);  // Original point cloud
    PointCloudT::Ptr cloud_tr (new PointCloudT);  // Transformed point cloud
    PointCloudT::Ptr cloud_icp (new PointCloudT);  // PICP output point cloud

    // Checking program arguments
    if(argc < 2){
        printf ("Usage :\n");
        printf ("\t\t%s file.ply number_of_PICP_iterations\n", argv[0]);
        PCL_ERROR ("Provide one ply file.\n");
        return (-1);
    }

    int iterations = 1;  // Default number of PICP iterations
    if (argc > 2){
        // If the user passed the number of iteration as an argument
        iterations = atoi (argv[2]);
        if (iterations < 1){
            PCL_ERROR ("Number of initial iterations must be >= 1\n");
            return (-1);
        }
    }

    if (pcl::io::loadPCDFile (argv[1], *cloud_in) < 0){
        PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }
    std::cout << "\nLoaded file " << argv[1] << " (" << cloud_in->size () << " points)" << std::endl;

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
    pcl::transformPointCloud(*cloud_in, *cloud_icp, transformation_matrix);
    *cloud_tr = *cloud_icp;

    // Gaussian noise to points in input clouds
    std::random_device rd{};
    std::mt19937 seed{rd()};
    double pcl_std_dev = 0.01;
    std::normal_distribution<double> d{0,pcl_std_dev};
    for(unsigned int i=0; i<cloud_in->points.size(); i++){
        cloud_in->points.at(i).x = cloud_in->points.at(i).x + d(seed);
        cloud_in->points.at(i).y = cloud_in->points.at(i).y + d(seed);
        cloud_in->points.at(i).z = cloud_in->points.at(i).z + d(seed);

        cloud_icp->points.at(i).x = cloud_icp->points.at(i).x + d(seed);
        cloud_icp->points.at(i).y = cloud_icp->points.at(i).y + d(seed);
        cloud_icp->points.at(i).z = cloud_icp->points.at(i).z + d(seed);
    }

    // PCL noise covariance
    Eigen::Matrix3f pcl_noise = Eigen::Matrix3f::Zero();
    pcl_noise(0,0) = std::pow(pcl_std_dev,2);
    pcl_noise(1,1) = std::pow(pcl_std_dev,2);
    pcl_noise(2,2) = std::pow(pcl_std_dev,2);

    // Apply initial (noisy) estimate of transform between trg and src point clouds
    double tf_std_dev = 0.6;
    std::normal_distribution<double> d2{0,tf_std_dev};
    theta = M_PI / 8 + d2(seed);
    transformation_matrix (0, 0) = cos (theta);
    transformation_matrix (0, 1) = sin (theta);
    transformation_matrix (1, 0) = -sin (theta);
    transformation_matrix (1, 1) = cos (theta);
    transformation_matrix (0, 3) += d2(seed);
    transformation_matrix (1, 3) += d2(seed);
    transformation_matrix (2, 3) += d2(seed);
    pcl::transformPointCloud(*cloud_icp, *cloud_icp, transformation_matrix);

    // Tf noise covariance
    Eigen::MatrixXf tf_noise = Eigen::MatrixXf::Zero(6,6);
    tf_noise(0,0) = std::pow(tf_std_dev,2);
    tf_noise(1,1) = std::pow(tf_std_dev,2);
    tf_noise(2,2) = std::pow(tf_std_dev,2);
    tf_noise(3,3) = 0.01;
    tf_noise(4,4) = 0.01;
    tf_noise(5,5) = 0.01;

    // Create pointclouds handlers
    pcl_map pcl_target, pcl_source;
    pcl_target.pcl_ = *cloud_in;
    pcl_source.pcl_ = *cloud_icp;

    // Add noise model to points and PCL poses
    for(unsigned int i=0; i<pcl_target.pcl_.size(); i++){
        pcl_target.pcl_covs_.push_back(pcl_noise);
    }

    for(unsigned int i=0; i<pcl_source.pcl_.size(); i++){
        pcl_source.pcl_covs_.push_back(pcl_noise);
    }
    pcl_source.cov_frame_ = tf_noise;
    pcl_target.cov_frame_ = tf_noise;

    // // Downsample preserving covariances attached
    // PointCloudT aux_pcl;
    // std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> aux_cov_vec;
    // unsigned int sub_cnt = 0;
    // // Target pcl
    // for(unsigned int i=0; i<pcl_target.pcl_.size(); i++){
    //     if(sub_cnt == 10){
    //         aux_pcl.points.push_back(pcl_target.pcl_.at(i));
    //         aux_cov_vec.push_back(pcl_target.pcl_covs_.at(i));
    //         sub_cnt = 0;
    //     }
    //     ++sub_cnt;
    // }
    // pcl_target.pcl_.clear();
    // pcl_target.pcl_ = aux_pcl;
    // pcl_target.pcl_covs_.clear();
    // pcl_target.pcl_covs_ = aux_cov_vec;

    // // Clear aux containers and repeat for src pcl
    // aux_pcl.clear();
    // aux_cov_vec.clear();
    // sub_cnt = 0;
    // for(unsigned int i=0; i<pcl_source.pcl_.size(); i++){
    //     if(sub_cnt == 10){
    //         aux_pcl.points.push_back(pcl_source.pcl_.at(i));
    //         aux_cov_vec.push_back(pcl_source.pcl_covs_.at(i));
    //         sub_cnt = 0;
    //     }
    //     ++sub_cnt;
    // }
    // pcl_source.pcl_.clear();
    // pcl_source.pcl_ = aux_pcl;
    // pcl_source.pcl_covs_.clear();
    // pcl_source.pcl_covs_ = aux_cov_vec;
    // aux_pcl.clear();
    // aux_cov_vec.clear();

    // Probabilistic ICP solver
    std::cout << "Creating PICP Solver" << std::endl;
    PointCloudT::Ptr smsrc_pcl_ptr (new PointCloudT);
    PointCloudT::Ptr smtrg_pcl_ptr (new PointCloudT);
    *smsrc_pcl_ptr = pcl_source.pcl_;
    *smtrg_pcl_ptr = pcl_target.pcl_;
    double lambda_thr = 20.6416; // Threshold for Mahalanobis distances in point matching
    boost::shared_ptr<ProbabilisticICP> icp_solver(new ProbabilisticICP(pcl_target, pcl_source, lambda_thr));


    // Initialize viewer object (use same while loop as in ICP PCL example?)
    pcl::visualization::PCLVisualizer viewer ("Probabilistic ICP demo");
    pclVisualizer(viewer, smtrg_pcl_ptr, smsrc_pcl_ptr, smsrc_pcl_ptr);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (smsrc_pcl_ptr, 180, 20, 20);

    // Construct KdTree for target pcl
    icp_solver->constructKdTree(pcl_target);
    bool converged = false;
    while (!viewer.wasStopped()){
        viewer.spinOnce ();
        // The user pressed "space" :
        if (next_iteration_icp){
            std::cout << "----------- Next iteration --------- \n" << std::endl;
            // The Iterative Closest Point algorithm
            icp_solver->alignStep(pcl_source);
            transformation_matrix = icp_solver->getTransformMatrix();
            print4x4Matrix(transformation_matrix);  // Print the transformation between original pose and current pose
            std::cout << "Current RMS error: " << icp_solver->getRMSError() << std::endl;

            // Update point cloud viewer
            smsrc_pcl_ptr.reset(new PointCloudT(pcl_source.pcl_));
            viewer.updatePointCloud(smsrc_pcl_ptr, cloud_icp_color_h, "cloud_icp_v2");
            converged = icp_solver->converged();
        }
        if(converged){
            std::cout << "------------------------------------------" << std::endl;
            std::cout << "Convergence! Final RMS error: " << icp_solver->getRMSError() << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            while (!viewer.wasStopped()){
                viewer.spinOnce ();
            }
            break;
        }
        next_iteration_icp = false;
    }
    return 0;
}
