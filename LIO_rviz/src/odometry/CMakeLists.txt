add_library(odometry STATIC
  vel_estimator.cpp
  pipeline.cpp
  mad_icp.cpp
)

target_include_directories(odometry PUBLIC
    #${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(odometry PUBLIC
  tools
  Eigen3::Eigen
)

install(TARGETS odometry
  LIBRARY DESTINATION lib
)

ament_export_libraries(odometry)
