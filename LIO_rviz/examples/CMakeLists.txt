# === GNSS-Inertial Odometry with UKF

add_executable(example_complete example_complete.cpp)
target_link_libraries(example_complete ukf_manifold_library)

add_executable(example_evaluation_metrics example_evaluation_metrics.cpp)
target_link_libraries(example_evaluation_metrics ukf_manifold_library)

add_executable(example_inizialization example_inizialization.cpp)
target_link_libraries(example_inizialization ukf_manifold_library)


# === LiDAR-Inertial Odometry with UKF & MAD-ICP

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${rosbag2_cpp_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS})

add_executable(example_UKF_NEWCO example_UKF_NEWCO.cpp)

target_link_libraries(example_UKF_NEWCO
  ukf_manifold_library
  odometry
  tools
  rclcpp::rclcpp
  rosbag2_cpp::rosbag2_cpp
  ${PCL_LIBRARIES})
  
ament_target_dependencies(example_UKF_NEWCO
  rclcpp
  sensor_msgs
  rosbag2_cpp
  pcl_conversions)
  
add_executable(LIO-UKF-MADICP LIO-UKF-MADICP.cpp)

target_link_libraries(LIO-UKF-MADICP
  ukf_manifold_library
  odometry
  tools
  rclcpp::rclcpp
  rosbag2_cpp::rosbag2_cpp
  ${PCL_LIBRARIES})
  
ament_target_dependencies(LIO-UKF-MADICP
  rclcpp
  sensor_msgs
  rosbag2_cpp
  pcl_conversions)
