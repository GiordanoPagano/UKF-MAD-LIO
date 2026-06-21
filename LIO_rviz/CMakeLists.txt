cmake_minimum_required(VERSION 3.8)
project(ukf_lio_rviz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_definitions(-DUKF_DATA_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_subdirectory(src)
add_subdirectory(examples)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

add_executable(lio_rviz ros_nodes/LIO_rviz.cpp)

target_link_libraries(lio_rviz
  ukf_manifold_library
  odometry
  tools
  ${PCL_LIBRARIES})
  
ament_target_dependencies(lio_rviz
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  rosbag2_cpp
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)

install(
    TARGETS lio_rviz
    DESTINATION lib/${PROJECT_NAME}
)


