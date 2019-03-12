#ifndef PICP_SOLVER_HPP
#define PICP_SOLVER_HPP

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <tuple>
#include <algorithm>
#include <functional>
#include <cmath>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Geometry>

#include "p_icp/utils.hpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class ProbabilisticICP{

public:

    ProbabilisticICP();

    ProbabilisticICP(pcl_map &submap_trg, pcl_map &submap_src, double delta_thr);

    void constructKdTree(const pcl_map &submap_trg);

    void alignStep(pcl_map &submap_src);

    Eigen::Matrix4f getTransformMatrix();

    double getRMSError();

    bool converged();

    void depictMatches(const PointCloudT::Ptr matches_trg, const PointCloudT::Ptr matches_src);

    double computeConvergence();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    // Inputs
    boost::shared_ptr<pcl_map> submap_ref_;
    Eigen::MatrixXf tf_noise_;

    // Estimated tf
    Eigen::Matrix4f tf_mat_;
    Eigen::Matrix4f tf_mat_prev_;


    // KdTree of target cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;

    // Center of mass of target cloud
    Eigen::Vector4f com_ref_;

    // Convergence error
    double rms_error_;
    double rms_error_prev_;
    Eigen::Matrix4f final_tf_mat_;

    // Aux
    double lambda_thr_;
    bool first_it_;
    bool converged_;
    PointCloudT::Ptr matches_trg_ptr_;
    PointCloudT::Ptr matches_src_ptr_;

    // Methods
    void computeTransformationP2P(const std::vector<icp_match> &matches_vec,
                                  const pcl_map &submap_src,
                                  Eigen::Matrix4f& transformation_matrix);

    void computeTransformationP2Plane(const std::vector<icp_match> &matches_vec, Eigen::Matrix4f &transformation_matrix);

    void applyState(Eigen::Matrix4f &t, const Eigen::VectorXf& x);

    std::vector<icp_match> point2PointAssoc(pcl_map &submap_src);

    std::vector<icp_match> point2PlaneAssoc(const pcl_map &submap_src);

    double computeRMSError(pcl_map &submap_src);

    void errorMinimization(const std::vector<icp_match> &matches_vec, Eigen::Matrix4f &transformation_matrix);

    Eigen::MatrixXf errorJacobian(PointT src_point, Eigen::Vector3f euler);

    Eigen::MatrixXf tfJacobian(Eigen::Vector3f point_i, Eigen::Vector3f euler);
};

#endif // PICP_SOLVER_HPP
