cmake_minimum_required(VERSION 3.8)
project(antigravity_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Detection messages
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  # Tracking messages
  "msg/TrackedObject.msg"
  "msg/TrackedObjectArray.msg"
  "msg/PredictedTrajectory.msg"
  # Semantic messages
  "msg/SemanticLabel.msg"
  "msg/SemanticMap.msg"
  # Localization messages
  "msg/LocalizationState.msg"
  "msg/ParticleCloud.msg"
  # Safety messages
  "msg/SafetyStatus.msg"
  "msg/SystemHealth.msg"
  "msg/GeofenceStatus.msg"
  # Planning messages
  "msg/WaypointPath.msg"
  "msg/TrajectoryPoint.msg"
  "msg/Trajectory.msg"
  "msg/RLAction.msg"
  # State estimation
  "msg/EKFState.msg"
  # Services
  "srv/LoadMap.srv"
  "srv/SetGoal.srv"
  "srv/GetLocalization.srv"
  "srv/SafetyOverride.srv"
  # Actions
  "action/NavigateToGoal.action"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
