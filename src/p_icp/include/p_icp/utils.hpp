#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>

#include "pcl/sample_consensus/sac_model.h"
#include "pcl/sample_consensus/model_types.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <eigen3/Eigen/Geometry>


using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointXYZ PointT;

// void getRemainingCorrespondences (
//     const pcl::Correspondences& original_correspondences, 
//     pcl::Correspondences& remaining_correspondences, 
//     PointCloudT::Ptr src_cloud,
//     PointCloudT::Ptr trg_cloud)
// {
//   int nr_correspondences = (int)original_correspondences.size ();
//   std::vector<int> source_indices (nr_correspondences);
//   std::vector<int> target_indices (nr_correspondences);

//   // Copy the query-match indices
//   for (size_t i = 0; i < original_correspondences.size (); ++i)
//   {
//     source_indices[i] = original_correspondences[i].index_query;
//     target_indices[i] = original_correspondences[i].index_match;
//   }

//    // from pcl/registration/icp.hpp:
//    std::vector<int> source_indices_good;
//    std::vector<int> target_indices_good;
//    {
//      // From the set of correspondences found, attempt to remove outliers
//      // Create the registration model
//      pcl::SampleConsensusModel<PointT>::Ptr model;
//     //  SampleConsensusModelNonRigidPtr model;
//      model.reset (new pcl::SampleConsensusModel<PointT> (src_cloud, source_indices));
//      // Pass the target_indices
//      model->setInputCloud (trg_cloud, target_indices);
//      // Create a RANSAC model
//      pcl::RandomSampleConsensus<PointT> sac (model, 0.05);
//      sac.setMaxIterations (1000);

//      // Compute the set of inliers
//      if (!sac.computeModel ())
//      {
//        remaining_correspondences = original_correspondences;
//     //    best_transformation_.setIdentity ();
//        return;
//      }
//      else
//      {
//        std::vector<int> inliers;
//        sac.getInliers (inliers);

//        if (inliers.size () < 3)
//        {
//          remaining_correspondences = original_correspondences;
//         //  best_transformation_.setIdentity ();
//          return;
//        }
//        std::unordered_map<int, int> index_to_correspondence;
//        for (int i = 0; i < nr_correspondences; ++i)
//          index_to_correspondence[original_correspondences[i].index_query] = i;

//        remaining_correspondences.resize (inliers.size ());
//        for (size_t i = 0; i < inliers.size (); ++i)
//          remaining_correspondences[i] = original_correspondences[index_to_correspondence[inliers[i]]];

//        // get best transformation
//        Eigen::VectorXf model_coefficients;
//        sac.getModelCoefficients (model_coefficients);
//     //    best_transformation_.row (0) = model_coefficients.segment<4>(0);
//     //    best_transformation_.row (1) = model_coefficients.segment<4>(4);
//     //    best_transformation_.row (2) = model_coefficients.segment<4>(8);
//     //    best_transformation_.row (3) = model_coefficients.segment<4>(12);
//      }
//    }
// }


struct pcl_map {
public:   
    int id_;
    PointCloudT pcl_;
    Eigen::Matrix4f rel_tf_;
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> pcl_covs_;
    Eigen::MatrixXf cov_frame_;   // 6x6 cov of submap frame
    unsigned int pings_num_;
    unsigned int beams_per_ping_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct icp_match{
public:

    icp_match(PointT trg_point, PointT src_point, Eigen::Vector3f normal, Eigen::Vector3f error_mean, Eigen::Matrix3f error_sigma){
        trg_point_ = trg_point;
        src_point_ = src_point;
        normal_ = normal;
        error_mean_ = error_mean;
        error_sigma_ = error_sigma;
    }

    PointT trg_point_;
    PointT src_point_;
    Eigen::Vector3f normal_;
    // Components of error pdf
    Eigen::Vector3f error_mean_;
    Eigen::Matrix3f error_sigma_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

vector<pair<int, int> > get_medgaz_matches();

void subsampleMap(pcl_map& pcl_map_in);

void outlierFilter(pcl_map& pcl_map_in);

Eigen::Vector3f computePCAPcl(PointCloudT& set_Ai);

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az);

Eigen::Matrix4f inverseTfMatrix(Eigen::Matrix4f tf_mat);

void plotSubmapsSet(const std::vector<pcl_map, Eigen::aligned_allocator<pcl_map> > &pcl_maps_set);


void print4x4Matrix (const Eigen::Matrix4f & matrix);

#endif // UTILS_HPP
