#include <p_icp/utils.hpp>

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Affine3d rx = Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry = Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz = Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}


Eigen::Matrix4f inverseTfMatrix(Eigen::Matrix4f tf_mat){

    Eigen::Matrix3f R_inv = tf_mat.topLeftCorner(3,3).transpose();

    Eigen::Matrix4f tf_mat_inv = Eigen::Matrix4f::Identity();
    tf_mat_inv.topLeftCorner(3,3) = R_inv;
    tf_mat_inv.topRightCorner(3,1) = R_inv * (-tf_mat.topRightCorner(3,1));

    return tf_mat_inv;

}


void plotSubmapsSet(const std::vector<pcl_map, Eigen::aligned_allocator<pcl_map>>& pcl_maps_set){
    // Plot submaps
    pcl::visualization::PCLVisualizer viewer ("ICP demo");
    viewer.setSize (1920, 1080);  // Visualiser window size

    PointCloudT::Ptr submap_i_ptr;
    int cnt = 0;
    for(const pcl_map& submap_i: pcl_maps_set){
//        if(submap_i.submap_id == 0 || submap_i.submap_id == 24){
            submap_i_ptr.reset(new PointCloudT(submap_i.pcl_));
            pcl::transformPointCloud(*submap_i_ptr, *submap_i_ptr, inverseTfMatrix(submap_i.rel_tf_));
            viewer.addPointCloud (submap_i_ptr, "cloud_" + std::to_string(cnt), 0);
//        }
        ++cnt;
    }

    while (!viewer.wasStopped()) {
        viewer.spinOnce ();
    }
}


void print4x4Matrix (const Eigen::Matrix4f & matrix) {
    printf ("Rotation matrix :\n");
    printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
    printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
    printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
    printf ("Translation vector :\n");
    printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}


vector<pair<int, int> > get_medgaz_matches()
{
    vector<pair<int, int> > matches;
    for (int i = 0; i < 3; ++i) {

        for (int j = 3; j < 5; ++j) {
            matches.push_back(make_pair(i, j));
        }

        for (int j = 5; j < 7; ++j) {
            matches.push_back(make_pair(i, j));
        }
    }
    return matches;
}


Eigen::Vector3f computePCAPcl(PointCloudT& set_Ai){

    // Compute the mean of the PCL
    Eigen::Vector4f com_Ai;
    pcl::compute3DCentroid(set_Ai, com_Ai);

    // Compute covariance matrix
    Eigen::Matrix3f cov_mat;
    pcl::computeCovarianceMatrixNormalized (set_Ai, com_Ai, cov_mat);
//    pcl::computeCovarianceMatrix(set_Ai_demean, cov_mat);

    // Extract eigenvalues and eigenvector from cov matrix
    Eigen::EigenSolver<Eigen::Matrix3f> eigenSolver;
    eigenSolver.compute(cov_mat, true);

    Eigen::MatrixXf eigenVectors = eigenSolver.eigenvectors().real();
    Eigen::VectorXf eigenvalues = eigenSolver.eigenvalues().real();

    // Return eigenvector with smallest eigenvalue
    std::vector<std::pair<double, int>> pidx;
    for (unsigned int i = 0 ; i<cov_mat.cols(); i++){
        pidx.push_back(std::make_pair(eigenvalues(i), i));
    }
    sort(pidx.begin(), pidx.end());

    return eigenVectors.col(pidx[0].second).transpose();
}


void subsampleMap(pcl_map& pcl_map_in){

    // Compute PCA of input pointcloud
    Eigen::Vector3f plane_normal = computePCAPcl(pcl_map_in.pcl_);

    // Create best fitting plane
    PointT sum_pointsj(0,0,0);
    double sum_traces = 0;
    double prj_trace;
    for(unsigned int i=0; i<pcl_map_in.pcl_.points.size(); i++){
        prj_trace = pcl_map_in.pcl_covs_.at(i).trace();
        sum_traces += std::pow(prj_trace, -2);
        sum_pointsj.getArray3fMap() += pcl_map_in.pcl_.points.at(i).getArray3fMap() * (float)std::pow(prj_trace, -2);
    }

    Eigen::Vector3f p_nu = Eigen::Vector3f(sum_pointsj.getArray3fMap() * 1/sum_traces);
    double d_p = plane_normal.dot(p_nu);

    // Distance of every point to their projection on the best fitting plane
    Eigen::Vector3f point_prj, distance;
    double average_dist = 0;
    for(const PointT& pointj: pcl_map_in.pcl_.points){
         point_prj =  Eigen::Vector3f(pointj.getArray3fMap()) - (Eigen::Vector3f(pointj.getArray3fMap()).dot(plane_normal) - d_p) * plane_normal;
         distance = Eigen::Vector3f(point_prj(0) - pointj.x,
                                    point_prj(1) - pointj.y,
                                    point_prj(2) - pointj.z);
         average_dist += distance.norm();
    }

    // Average of the distances
    average_dist = average_dist / pcl_map_in.pcl_.size();

    // Filter out points closer than average
    int cnt = 0;
    pcl_map pcl_map_aux;
    for(const PointT& pointj: pcl_map_in.pcl_.points){
        point_prj =  Eigen::Vector3f(pointj.getArray3fMap()) - (Eigen::Vector3f(pointj.getArray3fMap()).dot(plane_normal) - d_p) * plane_normal;
        distance = Eigen::Vector3f(point_prj(0) - pointj.x,
                                   point_prj(1) - pointj.y,
                                   point_prj(2) - pointj.z);
        if(distance.norm() >= average_dist*1.5){
            pcl_map_aux.pcl_.push_back(pointj);
            pcl_map_aux.pcl_covs_.push_back(pcl_map_in.pcl_covs_.at(cnt));
        }
        ++cnt;
    }

    // Clear and store final output values
    pcl_map_in.pcl_.clear();
    pcl_map_in.pcl_covs_.clear();

    pcl_map_in.pcl_ = pcl_map_aux.pcl_;
    pcl_map_in.pcl_covs_ = pcl_map_aux.pcl_covs_;
}


void outlierFilter(pcl_map &pcl_map_in){

    // Build Kdtree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    PointCloudT::Ptr pcl_ptr;
    pcl_ptr.reset(new PointCloudT(pcl_map_in.pcl_));
    kdtree.setInputCloud(pcl_ptr);

    // Average distance between NN points in pcl
    int average_nn = 0;
    int K = 4;
    std::vector<int> pointIdxRadiusSearch(K);
    std::vector<float> pointRadiusSquaredDistance(K);

    double radius = 1;
    for(PointT point_i: pcl_map_in.pcl_.points){
        average_nn += kdtree.radiusSearch (point_i, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    }
    average_nn = average_nn / pcl_map_in.pcl_.points.size();
    std::cout << "Average number of nn: " << average_nn << std::endl;

    // Filter out points farther from any other than average dist
    pcl_map pcl_map_aux;
    for(PointT point_i: pcl_map_in.pcl_.points){
        if(kdtree.radiusSearch (point_i, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) >= average_nn*0.2){
            pcl_map_aux.pcl_.push_back(point_i);
        }
    }

    pcl_map_in.pcl_.clear();
    pcl_map_in.pcl_ = pcl_map_aux.pcl_;
}

