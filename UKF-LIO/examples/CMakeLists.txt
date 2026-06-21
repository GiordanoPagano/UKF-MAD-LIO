# === GNSS-Inertial Odometry with UKF

add_executable(example_complete example_complete.cpp)
target_link_libraries(example_complete PRIVATE
    ukf_manifold
)

add_executable(example_evaluation_metrics example_evaluation_metrics.cpp)
target_link_libraries(example_evaluation_metrics  PRIVATE
    ukf_manifold
)
add_executable(example_inizialization example_inizialization.cpp)
target_link_libraries(example_inizialization  PRIVATE
    ukf_manifold
)

# === LiDAR-Inertial Odometry with UKF & MAD-ICP

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

#include_directories( 
#  # aggiunge variabili globali che possono essere viste da tutti gli eseguibili; 
#  # vecchio stile, da aggiornare

#  ${rclcpp_INCLUDE_DIRS}
#  ${sensor_msgs_INCLUDE_DIRS}
#  ${rosbag2_cpp_INCLUDE_DIRS}
#  ${PCL_INCLUDE_DIRS}
#)

add_executable(example_UKF_NEWCO example_UKF_NEWCO.cpp)

target_link_libraries(example_UKF_NEWCO  PRIVATE
  ukf_manifold
  odometry
  tools
  ${PCL_LIBRARIES}
)
  
target_include_directories(example_UKF_NEWCO PRIVATE
    ${PCL_INCLUDE_DIRS}
)
  
# con questo comando si aggiungono sia librerie che directories,
# quindi non serve più dichiararle globali
# non può essere usato con PCL_INCLUDE_DIRS/PCL_LIBRARIES perché è una libreria vecchia
ament_target_dependencies(example_UKF_NEWCO  
  rclcpp
  sensor_msgs
  rosbag2_cpp
  pcl_conversions)
  
add_executable(LIO-UKF-MADICP LIO-UKF-MADICP.cpp)

target_link_libraries(LIO-UKF-MADICP PRIVATE
  ukf_manifold
  odometry
  tools
  ${PCL_LIBRARIES}
)
  
target_include_directories(LIO-UKF-MADICP PRIVATE
    ${PCL_INCLUDE_DIRS}
)
  
ament_target_dependencies(LIO-UKF-MADICP
  rclcpp
  sensor_msgs
  rosbag2_cpp
  pcl_conversions)
