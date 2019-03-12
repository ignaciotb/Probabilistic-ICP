project(p_icp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

include_directories(../p_icp/include/)


add_executable(test_picp
    src/test_picp.cpp
    src/utils.cpp
    src/picp_solver.cpp
)

target_link_libraries(test_picp
    ${PCL_LIBRARIES}
)
