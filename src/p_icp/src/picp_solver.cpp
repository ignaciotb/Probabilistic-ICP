#include <p_icp/picp_solver.hpp>

ProbabilisticICP::ProbabilisticICP(){

}


ProbabilisticICP::ProbabilisticICP(pcl_map &submap_trg, pcl_map &submap_src, double delta_thr){

    submap_ref_.reset(new pcl_map(submap_trg));
    pcl::compute3DCentroid(submap_ref_->pcl_, com_ref_);

    std::cout << "Size of src: " << submap_src.pcl_.size() << std::endl;
    std::cout << "Size of trg: " << submap_ref_->pcl_.size() << std::endl;

    // Compute relative tf noise
    tf_noise_ = submap_src.cov_frame_; //+ submap_trg.cov_frame_; //TODO: how to do this?

    // Create Xi squared dist to filter out outliers in matching
//    boost::math::chi_squared chi2_dist(3);
    lambda_thr_ = delta_thr; // boost::math::quantile(chi2_dist, delta_thr);
    first_it_ = true;
    final_tf_mat_.setIdentity();
    converged_ = false;

    // Construct KdTree for target pcl
    this->constructKdTree(submap_trg);
    std::cout << "Kd tree built" << std::endl;

    // Initial error between pointclouds
    rms_error_prev_ = computeRMSError(submap_src);
    std::cout << "Initial RMS Error: " << rms_error_prev_ << std::endl;

}


Eigen::Matrix4f ProbabilisticICP::getTransformMatrix(){
    return final_tf_mat_;
}


double ProbabilisticICP::getRMSError(){
    return rms_error_;
}


void ProbabilisticICP::constructKdTree(const pcl_map& submap_trg){
    // Construct Kd Tree with target cloud
    PointCloudT::Ptr pcl_ptr;
    pcl_ptr.reset(new PointCloudT(submap_trg.pcl_));
    kdtree_.setInputCloud(pcl_ptr);
}


bool ProbabilisticICP::converged(){
    return converged_;
}


void ProbabilisticICP::alignStep(pcl_map &submap_src){

    // Find matches tuples
    std::vector<icp_match> matches_vec;
    tf_mat_prev_ = tf_mat_; // Store previous transform
    matches_vec = point2PlaneAssoc(submap_src);
    computeTransformationP2Plane(matches_vec, tf_mat_);

    // Transform cloud_new based on latest tf
    pcl::transformPointCloud(submap_src.pcl_, submap_src.pcl_, tf_mat_);

    // Root Mean Square Error to measure convergence
    rms_error_ = computeRMSError(submap_src);

    // Check convergence
    if(rms_error_prev_ > rms_error_){
        final_tf_mat_ = final_tf_mat_ * tf_mat_;
    }
    else{
        converged_ = true;
    }

    rms_error_prev_ = rms_error_;
}


Eigen::MatrixXf ProbabilisticICP::tfJacobian(Eigen::Vector3f point_i, Eigen::Vector3f euler){

    // Transform jacobian
    Eigen::MatrixXf jacobian_tf = Eigen::MatrixXf::Zero(3,6);
    jacobian_tf.topLeftCorner(3,3) = Eigen::Matrix3f::Identity();

    // Rotation matrix
    Eigen::Matrix3f rotZ;
    rotZ << 1,             0,              0,
            0, cos(euler(2)), -sin(euler(2)),
            0, sin(euler(2)),  cos(euler(2));
    Eigen::Matrix3f rotY;
    rotY << cos(euler(1)),  0, sin(euler(1)),
                        0,  1,             0,
            -sin(euler(1)), 0, cos(euler(1));
    Eigen::Matrix3f rotX;
    rotX << cos(euler(0)), -sin(euler(0)),  0,
            sin(euler(0)),  cos(euler(0)),  0,
                        0,              0,  1;

    // Derivatives of rotations
    Eigen::Matrix3f dev_rotZ;
    dev_rotZ << 1,              0,               0,
                0, -sin(euler(2)),  -cos(euler(2)),
                0,  cos(euler(2)),  -sin(euler(2));
    Eigen::Matrix3f dev_rotY;
    dev_rotY << -sin(euler(1)),  0,  cos(euler(1)),
                             0,  1,              0,
                -cos(euler(1)),  0, -sin(euler(1));
    Eigen::Matrix3f dev_rotX;
    dev_rotX << -sin(euler(0)),  -cos(euler(0)), 0,
                 cos(euler(0)),  -sin(euler(0)), 0,
                             0,               0, 1;

    jacobian_tf.col(3) = dev_rotZ * rotY * rotX * point_i;
    jacobian_tf.col(4) = rotZ * dev_rotY * rotX * point_i;
    jacobian_tf.col(5) = rotZ * rotY * dev_rotX * point_i;

    return jacobian_tf;

}

std::vector<icp_match> ProbabilisticICP::point2PlaneAssoc(const pcl_map& submap_src){

    using namespace std;
//    std::cout << "Point to plane association" << std::endl;

    matches_trg_ptr_.reset(new PointCloudT);
    matches_src_ptr_.reset(new PointCloudT);

    // K nearest neighbor search
    int K = 30;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    // Point-point probabilistic association
    Eigen::Vector3f error_mean;
    double pow_mhl_dist;
    PointCloudT set_AiPCL;
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> set_AiCovs;

    // Covariance matrix of error distribution: since pcls and tf have const covariances, it can be computed only once
    Eigen::Matrix3f error_cov_p2plane;
    Eigen::MatrixXf jacobian_tf;

    // For every point in transformed pcl
    std::vector<icp_match> matches_vec;
    int point_cnt = 0;

//    for(unsigned int col = 0; col < submap_src.pings_num_; col++){
//        for(unsigned int row = 0; row < submap_src.beams_per_ping_; row++){

    for(PointT point_ni: submap_src.pcl_.points){
        // Find kNN
        if(kdtree_.nearestKSearch(point_ni, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){

            // For every potential nearest neighbor
            for(int pointTrgId: pointIdxNKNSearch){
                // Mean and cov mat of error distribution
                error_mean = Eigen::Vector3f(submap_ref_->pcl_.points[pointTrgId].x - point_ni.x,
                                             submap_ref_->pcl_.points[pointTrgId].y - point_ni.y,
                                             submap_ref_->pcl_.points[pointTrgId].z - point_ni.z);

//                jacobian_tf = submap_src.auv_pose_.at(point_cnt);

                jacobian_tf = this->tfJacobian(Eigen::Vector3f(point_ni.x, point_ni.y, point_ni.z), Eigen::Vector3f(0,0,0.2));
                error_cov_p2plane = submap_src.pcl_covs_.at(point_cnt) + jacobian_tf * tf_noise_ * jacobian_tf.transpose() + submap_ref_->pcl_covs_.at(pointTrgId);

                // Mahalanobis distance
                pow_mhl_dist = error_mean.transpose() * error_cov_p2plane.inverse() * error_mean;

                // If Mhl dist smaller than Xi squared threshold, add to set of compatible points
                if(pow_mhl_dist < lambda_thr_){
                    set_AiPCL.points.push_back(submap_ref_->pcl_.points[pointTrgId]);
                    set_AiCovs.push_back(submap_ref_->pcl_covs_.at(pointTrgId));
                }
            }

            // Point to Plane association: fit a plane to the points in set_Ai
            if(set_AiPCL.points.size() >= 3){

                // Weighted center of set Ai
                PointT sum_pointsj(0,0,0);
                double sum_traces = 0;
                double prj_trace;
                for(unsigned int i=0; i<set_AiPCL.points.size(); i++){
                    prj_trace = set_AiCovs.at(i).trace();
                    sum_traces += std::pow(prj_trace, -2);
                    sum_pointsj.getArray3fMap() += set_AiPCL.points.at(i).getArray3fMap() * (float)std::pow(prj_trace, -2);
                }
                Eigen::Vector3f p_nu = Eigen::Vector3f(sum_pointsj.getArray3fMap() * 1/sum_traces);

                // Run PCA on set Ai to define normal vector to plane
                Eigen::Vector3f normal_vec = computePCAPcl(set_AiPCL);

                // Calculate distance d_p origin to plane
                double d_p = normal_vec.dot(p_nu);

                // Orthogonal projection of point_ni over the plane fitted
                Eigen::Vector3f point_ai =  Eigen::Vector3f(point_ni.getArray3fMap()) - (Eigen::Vector3f(point_ni.getArray3fMap()).dot(normal_vec) - d_p) * normal_vec;

                // Error distribution between ai and ni
                // Mean and cov mat of error distribution
                error_mean = Eigen::Vector3f(point_ai(0) - point_ni.x,
                                             point_ai(1) - point_ni.y,
                                             point_ai(2) - point_ni.z);

                // Mahalanobis distance
//                error_cov_p2plane = submap_src.pcl_covs_.at(point_cnt) + jacobian_tf * tf_noise_ * jacobian_tf.transpose();   // For point to plane
                pow_mhl_dist = error_mean.transpose() * error_cov_p2plane.inverse() * error_mean;

                // If Mhl under threshold, create match (ai, ni)
                if(pow_mhl_dist < lambda_thr_){
                    matches_vec.push_back(icp_match(PointT(point_ai(0), point_ai(1), point_ai(2)),
                                                         point_ni,
                                                         normal_vec,
                                                         error_mean,
                                                         error_cov_p2plane));
                    // Store points matched for viewers
                    matches_trg_ptr_->points.push_back(PointT(point_ai(0), point_ai(1), point_ai(2)));
                    matches_src_ptr_->points.push_back(point_ni);
                }

                set_AiPCL.points.clear();
            }
        }
        ++point_cnt;
    }

    std::cout<< "Size of matches vector " << matches_vec.size() << std::endl;
    return matches_vec;
}


void ProbabilisticICP::computeTransformationP2Plane(const std::vector<icp_match> &matches_vec,
                                                    Eigen::Matrix4f &transformation_matrix){

    using namespace Eigen;
    using namespace std;

    // Construct linear system A*tau = b
    MatrixXf mat_A = MatrixXf::Zero(6,6);
    Eigen::VectorXf vec_b = Eigen::VectorXf::Zero(6);
    Eigen::VectorXf tau = Eigen::VectorXf(6);

    // Aux vectors
    Vector3f c_k;
    Vector3f d_k;
    VectorXf vec_joined(6);
    Vector3f n_k;
    for(const icp_match match: matches_vec){
        n_k = match.normal_;
        // Cross product of point in source  and normal to surface in reference
        c_k = Vector3f(match.src_point_.x, match.src_point_.y, match.src_point_.z).cross(n_k);
        // Reference minus source point
        d_k = Vector3f(match.trg_point_.x, match.trg_point_.y, match.trg_point_.z) - Vector3f(match.src_point_.x, match.src_point_.y, match.src_point_.z);
        vec_joined << c_k, n_k;
        // Matrix A as a summation of the vec_joined * vec_joined'
        mat_A += vec_joined * vec_joined.transpose();
        // Vector b as a summation of vec_joined * (d_k * normal_k)
        vec_b += vec_joined * d_k.dot(n_k);
    }

    // Solve linear system for tau with JacobiSVD
    tau = mat_A.fullPivLu().solve(vec_b);

    // Compute transformation matrix from tau solution
    transformation_matrix = Eigen::Matrix4f::Identity();
    applyState(transformation_matrix, tau);
}


void ProbabilisticICP::applyState(Eigen::Matrix4f &t, const Eigen::VectorXf& x){
  // !!! CAUTION Stanford GICP uses the Z Y X euler angles convention
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf (x[2], Eigen::Vector3f::UnitZ ())
    * Eigen::AngleAxisf (x[1], Eigen::Vector3f::UnitY ())
    * Eigen::AngleAxisf (x[0], Eigen::Vector3f::UnitX ());
  t.topLeftCorner<3,3> ().matrix () = R * t.topLeftCorner<3,3> ().matrix ();
  Eigen::Vector4f T (x[3], x[4], x[5], 0.0f);
  t.col (3) += T;
}


std::vector<icp_match> ProbabilisticICP::point2PointAssoc(pcl_map& submap_src){

    std::cout << "Point to point association" << std::endl;

    matches_trg_ptr_.reset(new PointCloudT);
    matches_src_ptr_.reset(new PointCloudT);

    // K nearest neighbor search
    int K = 30;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    // Point-point probabilistic association
    Eigen::Vector3f error_mean;
    Eigen::Matrix3f error_cov;
    double pow_mhl_dist;
    std::vector<int> set_Ai;

    // TODO: project tf_noise_ to pcl_noise subspace with the appropiate jacobian
    // https://www.frc.ri.cmu.edu/~hpm/project.archive/reference.file/Smith,Self&Cheeseman.pdf page 176
    Eigen::MatrixXf jacobian_1 = Eigen::MatrixXf::Zero(3,6);
    jacobian_1.topLeftCorner(3,3) = Eigen::Matrix3f::Identity();
    jacobian_1.topRightCorner(3,3) = Eigen::Matrix3f::Identity();

    // For every point in transformed pcl
    std::vector<icp_match> matches_vec;
    std::vector<int>::iterator min_match_it;
    unsigned int point_cnt = 0;
    // Probabilistic P2P assoc
    for(PointT point_ni: submap_src.pcl_.points){
        if(kdtree_.nearestKSearch(point_ni, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
            // For every potential nearest neighbor
            for(int pointTrgId: pointIdxNKNSearch){
                // Mean and cov mat of error distribution
                error_mean = Eigen::Vector3f(submap_ref_->pcl_.points[pointTrgId].x - point_ni.x,
                                             submap_ref_->pcl_.points[pointTrgId].y - point_ni.y,
                                             submap_ref_->pcl_.points[pointTrgId].z - point_ni.z);

                error_cov = submap_src.pcl_covs_.at(point_cnt) + jacobian_1 * tf_noise_ * jacobian_1.transpose() + submap_ref_->pcl_covs_.at(pointTrgId);

                // Mahalanobis distance
                pow_mhl_dist = error_mean.transpose() * error_cov.inverse() * error_mean;

                // If Mhl dist smaller than Xi squared threshold, add to set of compatible points
                if(pow_mhl_dist < lambda_thr_){
                    set_Ai.push_back(pointTrgId);
                }
            }
            // The match with smallest Mhl distance is selected
            if(!set_Ai.empty()){
                min_match_it = std::min_element(std::begin(set_Ai), std::end(set_Ai));
                // Add zeros vector when a normal has not been computed
                matches_vec.push_back(icp_match(submap_ref_->pcl_.points[set_Ai.at(std::distance(std::begin(set_Ai), min_match_it))],
                                                point_ni,
                                                Eigen::Vector3f(0,0,0),
                                                error_mean,
                                                error_cov));

                // Store points matched for viewers
                matches_trg_ptr_->points.push_back(submap_ref_->pcl_.points[set_Ai.at(std::distance(std::begin(set_Ai), min_match_it))]);
                matches_src_ptr_->points.push_back(point_ni);
                set_Ai.clear();
            }
        }
        ++point_cnt;
    }
    // End prob P2P

    // P2P assoc
//    double dmax_squared = std::pow(1, 2);
//    for(PointT point_i: submap_src.pcl_.points){
//        if(kdtree_.nearestKSearch(point_i, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
//            if(pointNKNSquaredDistance[0] <= dmax_squared){
//                matches_vec.push_back(std::make_tuple(submap_ref_->pcl_.points[pointIdxNKNSearch[0]], point_i, Eigen::Vector3f(0,0,0)));
//            }
//        }
//    }
    // End plain P2P assoc

    std::cout<< "Size of matches vector " << matches_vec.size() << std::endl;

    return matches_vec;
}


void ProbabilisticICP::computeTransformationP2P(const std::vector<icp_match>& matches_vec,
                                                const pcl_map &submap_src,
                                                Eigen::Matrix4f& transformation_matrix){

    // Center of mass of tf point cloud
    Eigen::Vector4f com_tf;
    pcl::compute3DCentroid(submap_src.pcl_, com_tf);

    // Demean all points in the matches
    Eigen::MatrixXf trg_demean = Eigen::MatrixXf::Zero(3, matches_vec.size());
    Eigen::MatrixXf tf_demean = Eigen::MatrixXf::Zero(matches_vec.size(), 3);

    unsigned int match_cnt = 0;
    for(icp_match match: matches_vec){
        trg_demean.col(match_cnt) = Eigen::Vector3f(match.trg_point_.x,
                                                    match.trg_point_.y,
                                                    match.trg_point_.z) - com_ref_.head(3);

        tf_demean.row(match_cnt) = (Eigen::Vector3f(match.src_point_.x,
                                                    match.src_point_.y,
                                                    match.src_point_.z) - com_tf.head(3)).transpose();

        match_cnt += 1;
    }

    // Assemble the correlation matrix H = source * target'
    Eigen::Matrix3f H = trg_demean * tf_demean;

    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Matrix3f v = svd.matrixV();

    // Compute R = U * V'
    if(u.determinant() * v.determinant() < 0) {
        for(int x = 0; x < 3; ++x) {
            v(x, 2) *= -1;
        }
    }
    Eigen::Matrix3f R = u * v.transpose();

    // Return the correct transformation
    transformation_matrix.setIdentity();

    transformation_matrix.topLeftCorner(3, 3) = R;
    const Eigen::Vector3f Rc(R * com_tf.head(3));
    transformation_matrix.block(0, 3, 3, 1) = com_ref_.head(3) - Rc;

}

// From PCL GICP
double ProbabilisticICP::computeConvergence(){

    /* compute the delta from this iteration */
    double rotation_epsilon_ = 2e-3;
    double transformation_epsilon_ = 5e-2;

    double delta = 0.;
    for(int k = 0; k < 4; k++) {
      for(int l = 0; l < 4; l++) {
        double ratio = 1;
        if(k < 3 && l < 3) // rotation part of the transform
          ratio = 1./rotation_epsilon_;
        else
          ratio = 1./transformation_epsilon_;
        double c_delta = ratio*fabs(tf_mat_prev_(k,l) - tf_mat_(k,l));
        if(c_delta > delta)
          delta = c_delta;
      }
    }

    return delta;
}

double ProbabilisticICP::computeRMSError(pcl_map& submap_src){

    // Find matches tuples
    std::vector<icp_match> matches_vec = point2PlaneAssoc(submap_src);

    // Compute RMS Error
    double rmsError = 0;
    PointT diff;
    for(icp_match match: matches_vec){
        diff.getArray3fMap() = match.trg_point_.getArray3fMap() - match.src_point_.getArray3fMap();
        rmsError += Eigen::Vector3f(diff.x, diff.y, diff.z).norm();
    }

    return std::sqrt(rmsError / matches_vec.size());
}


void ProbabilisticICP::depictMatches(const PointCloudT::Ptr matches_trg, const PointCloudT::Ptr matches_src){
    pcl::visualization::PCLVisualizer viewerMono ("ICP matches");

    // Create two vertically separated viewports
    int v1 (0);

    // The color we will be using
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (matches_trg, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                               (int) 255 * txt_gray_lvl);
    viewerMono.addPointCloud (matches_trg, cloud_in_color_h, "cloud_in_v1", v1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (matches_src, 20, 180, 20);
    viewerMono.addPointCloud (matches_src, cloud_tr_color_h, "cloud_tr_v1", v1);

    // Set background color
    viewerMono.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);

    // Set camera position and orientation
    viewerMono.setCameraPosition (-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewerMono.setSize (1280, 1024);  // Visualiser window size

    while (!viewerMono.wasStopped()){
        viewerMono.spinOnce ();
    }
}

void ProbabilisticICP::errorMinimization(const std::vector<icp_match> &matches_vec, Eigen::Matrix4f &transformation_matrix){

    using namespace Eigen;
    unsigned int matches_num = (11000 > matches_vec.size())? matches_vec.size(): 11000;
//    int matches_num = (int)matches_vec.size();
    std::cout << "Number of matches to sample " << matches_num << std::endl;

    // Extract Euler angles from tf matrix
    Matrix3f R = transformation_matrix.topLeftCorner(3,3);
    Vector3f euler = R.eulerAngles(0, 1, 2);

    MatrixXf J(3*matches_num,6);
    VectorXf error_vec(3*matches_num);
    MatrixXf P_error_inv = MatrixXf::Zero(3*matches_num, 3*matches_num);
    std::cout << "Matrices allocated " << std::endl;

    unsigned int cnt = 0;
    boost::shared_ptr<icp_match> match;
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    unsigned int x;
    for(unsigned int i = 0; i < matches_num; i++){
        x = matches_vec.size()+1;
        while(x >= matches_vec.size())
            x = std::rand()/((RAND_MAX)/matches_vec.size());  // Note: 1+rand()%6 is biased
//        std::cout << "idx " << x << std::endl;
        match.reset(new icp_match(matches_vec.at(x)));
        // Create Jacobian of error function evaluated in all points
        J.block(cnt*3, 0, 3, 6) = errorJacobian(match->src_point_, euler);
        // Construct block-diagonal covariance matrix
        P_error_inv.block(cnt*3, cnt*3, 3, 3) = match->error_sigma_.inverse();
        // Concatenate error vectors
        error_vec.segment(cnt*3, 3) = match->error_mean_;
        ++cnt;
    }

    // Compute transform components
    std::cout << "Calculating LS Solution" << std::endl;
    VectorXf tau(6);
    MatrixXf aux = J.transpose() * P_error_inv;
    MatrixXf A_mat  = aux * J;
    VectorXf b_vec = aux * error_vec;

    // Solve for tau with BDCSVD
    tau = A_mat.bdcSvd(ComputeThinU | ComputeThinV).solve(b_vec);
//    tau = A_mat.fullPivLu().solve(b_vec);

    // Construct transformation matrix from tf components
    VectorXf tau_adapt(6);
    tau_adapt.segment(0,3) = tau.segment(3,3);
    tau_adapt.segment(3,3) = tau.segment(0,3);
    transformation_matrix = Eigen::Matrix4f::Identity();
    applyState(transformation_matrix, tau_adapt);
}


Eigen::MatrixXf ProbabilisticICP::errorJacobian(PointT src_point, Eigen::Vector3f euler){

    using namespace std;
    using namespace Eigen;

    MatrixXf jacobian(3,6);
    Vector3f src_p = Vector3f(src_point.x, src_point.y, src_point.z);

    jacobian(0,0) = - cos(euler(1))*cos(euler(2)) - 1;
    jacobian(0,1) = cos(euler(0))*sin(euler(2)) - cos(euler(2))*sin(euler(1))*sin(euler(0));
    jacobian(0,2) = -sin(euler(0))*sin(euler(2)) - cos(euler(0))*cos(euler(2))*sin(euler(1));
    jacobian(0,3) = -src_p(1)*(sin(euler(0))*sin(euler(2)) + cos(euler(0))*cos(euler(2))*sin(euler(1))) - src_p(2)*(cos(euler(0))*sin(euler(2)) - cos(euler(2))*sin(euler(1))*sin(euler(0)));
    jacobian(0,4) = -cos(euler(2))*(src_p(2)*cos(euler(1))*cos(euler(0)) - src_p(0)*sin(euler(1)) + src_p(1)*cos(euler(1))*sin(euler(0)));
    jacobian(0,5) = src_p(1)*(cos(euler(0))*cos(euler(2)) + sin(euler(1))*sin(euler(0))*sin(euler(2))) - src_p(2)*(cos(euler(2))*sin(euler(0)) - cos(euler(0))*sin(euler(1))*sin(euler(2))) + src_p(0)*cos(euler(1))*sin(euler(2));

    jacobian(1,0) = -cos(euler(1))*sin(euler(2));
    jacobian(1,1) = -cos(euler(0))*cos(euler(2)) - sin(euler(1))*sin(euler(0))*sin(euler(2)) - 1;
    jacobian(1,2) = cos(euler(2))*sin(euler(0)) - cos(euler(0))*sin(euler(1))*sin(euler(2));
    jacobian(1,3) = src_p(1)*(cos(euler(2))*sin(euler(0)) - cos(euler(0))*sin(euler(1))*sin(euler(2))) + src_p(2)*(cos(euler(0))*cos(euler(2)) + sin(euler(1))*sin(euler(0))*sin(euler(2)));
    jacobian(1,4) = -sin(euler(2))*(src_p(2)*cos(euler(1))*cos(euler(0)) - src_p(0)*sin(euler(1)) + src_p(1)*cos(euler(1))*sin(euler(0)));
    jacobian(1,5) = src_p(1)*(cos(euler(0))*sin(euler(2)) - cos(euler(2))*sin(euler(1))*sin(euler(0))) - src_p(2)*(sin(euler(0))*sin(euler(2)) + cos(euler(0))*cos(euler(2))*sin(euler(1))) - src_p(0)*cos(euler(1))*cos(euler(2));

    jacobian(2,0) = sin(euler(1));
    jacobian(2,1) = -cos(euler(1))*sin(euler(0));
    jacobian(2,2) = -cos(euler(1))*cos(euler(0)) - 1;
    jacobian(2,3) = -cos(euler(1))*(src_p(1)*cos(euler(0)) - src_p(2)*sin(euler(0)));
    jacobian(2,4) = src_p(0)*cos(euler(1)) + src_p(2)*cos(euler(0))*sin(euler(1)) + src_p(1)*sin(euler(1))*sin(euler(0));
    jacobian(2,5) = 0;

    return -1*jacobian;
}



