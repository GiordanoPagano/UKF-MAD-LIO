cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ukf_manifold LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall")  #-Werror


find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)

add_definitions(-DUKF_EXAMPLE_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/examples")
add_definitions(-DUKF_DATA_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

message("-- ${PROJECT_NAME}|example dir = ${CMAKE_CURRENT_SOURCE_DIR}/examples")
message("-- ${PROJECT_NAME}|data dir    = ${CMAKE_CURRENT_SOURCE_DIR}/test_data")


add_subdirectory(src)
add_subdirectory(examples)
