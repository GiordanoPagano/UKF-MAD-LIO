cmake_minimum_required(VERSION 3.16)

project(ukf_manifold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_subdirectory(src)

add_subdirectory(examples)
