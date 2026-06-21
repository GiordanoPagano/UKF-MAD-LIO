add_library(odometry STATIC
  vel_estimator.cpp
  pipeline.cpp
  mad_icp.cpp
)

find_package(OpenMP)

target_link_libraries(odometry PUBLIC
  tools
  Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(odometry PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_features(odometry PUBLIC cxx_std_17)

