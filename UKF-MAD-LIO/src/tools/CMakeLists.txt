add_library(tools STATIC
    mad_tree.cpp
)

target_include_directories(tools PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(tools PUBLIC
    Eigen3::Eigen
)

target_compile_features(tools PUBLIC cxx_std_17)
