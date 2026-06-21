add_library(tools STATIC
  mad_tree.cpp
)

target_include_directories(tools PUBLIC
    #${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(tools PUBLIC Eigen3::Eigen)

ament_target_dependencies(tools Eigen3)

install(TARGETS tools
  LIBRARY DESTINATION lib
)

ament_export_libraries(tools)
