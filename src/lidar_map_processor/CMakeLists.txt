cmake_minimum_required(VERSION 3.8)
project(lidar_map_processor)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)

# ONNX Runtime paths
#set(ONNXRUNTIME_ROOT "/home/yuanyan/onnxruntime-1.15.1")
#set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
#set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

add_executable(lidar_map_processor src/lidar_map_processor.cpp)
target_include_directories(lidar_map_processor PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR} # Include ONNX Runtime headers
  )
  
target_compile_features(lidar_map_processor PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  lidar_map_processor
  "pcl_conversions"
  "pcl_msgs"
  "octomap_msgs"
  "rclcpp"
  "OpenCV"
)
# Link libraries
#target_link_libraries(lidar_map_processor
#  ${ONNXRUNTIME_LIB}         # Link ONNX Runtime library
#)

install(TARGETS lidar_map_processor
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
