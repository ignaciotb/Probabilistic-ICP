#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <eigen3/Eigen/Geometry>


using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointXYZ PointT;


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
