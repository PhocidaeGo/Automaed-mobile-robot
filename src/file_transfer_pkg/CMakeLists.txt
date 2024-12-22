cmake_minimum_required(VERSION 3.5)
project(file_transfer_pkg)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Executables
add_executable(file_sender src/file_sender.cpp)
ament_target_dependencies(file_sender rclcpp std_msgs)

add_executable(file_receiver src/file_receiver.cpp)
ament_target_dependencies(file_receiver rclcpp std_msgs)

# Install
install(TARGETS
  file_sender
  file_receiver
  DESTINATION lib/${PROJECT_NAME})

ament_package()
