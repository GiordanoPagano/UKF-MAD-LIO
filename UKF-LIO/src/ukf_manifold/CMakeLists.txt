target_compile_features(ukf_manifold_library PUBLIC)

add_library(ukf_manifold
    nav_state.cpp
)

target_include_directories(ukf_manifold_library PUBLIC
    #${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(ukf_manifold_library PUBLIC
    Eigen3::Eigen
)
