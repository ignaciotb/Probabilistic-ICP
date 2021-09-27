project(p_icp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

# include_directories(../p_icp/include/)

add_executable (test_narf 
	src/test_narf.cpp)
target_link_libraries (test_narf 
	${PCL_LIBRARIES})

add_executable (test_features
	src/test_features.cpp)
target_link_libraries (test_features 
	${PCL_LIBRARIES})

add_executable (test_sift_normals
	src/test_sift_normals.cpp)
target_link_libraries (test_sift_normals
	${PCL_LIBRARIES})

add_executable (test_sift_depth
	src/test_sift_depth.cpp)
target_link_libraries (test_sift_depth
	${PCL_LIBRARIES})

add_executable (test_full_corr
	src/test_full_corr.cpp)
target_link_libraries (test_full_corr
	${PCL_LIBRARIES})
