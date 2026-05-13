cmake_minimum_required(VERSION 3.8)
project(antigravity_bringup)

find_package(ament_cmake REQUIRED)

# Install launch files, configs, RViz, URDF, and worlds
install(DIRECTORY
  launch
  config
  rviz
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
