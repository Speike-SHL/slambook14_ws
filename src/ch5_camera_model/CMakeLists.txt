cmake_minimum_required(VERSION 3.5)
project(ch5_camera_model)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)
find_package(FMT REQUIRED)

include_directories(${Sophus_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})

# 编译设置
add_executable(5.3.1_opencvBasic src/5.3.1_opencvBasic.cpp)
target_link_libraries(5.3.1_opencvBasic ${OpenCV_LIBS})
target_compile_options(5.3.1_opencvBasic PRIVATE -Wno-unused-variable)

add_executable(5.3.2_undistortImage src/5.3.2_undistortImage.cpp)
target_link_libraries(5.3.2_undistortImage ${OpenCV_LIBS})

add_executable(5.4.1_stereoVision_node src/5.4.1_stereoVision_node.cpp)
target_link_libraries(5.4.1_stereoVision_node ${OpenCV_LIBS} )
ament_target_dependencies(5.4.1_stereoVision_node
  rclcpp
  sensor_msgs
)

add_executable(5.4.2_rgbdJointMap_node src/5.4.2_rgbdJointMap_node.cpp)
target_link_libraries(5.4.2_rgbdJointMap_node ${OpenCV_LIBS} ${PCL_LIBRARIES} fmt::fmt)
ament_target_dependencies(5.4.2_rgbdJointMap_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

# install settings
install(TARGETS
  5.3.1_opencvBasic
  5.3.2_undistortImage
  5.4.1_stereoVision_node
  5.4.2_rgbdJointMap_node
  DESTINATION lib/${PROJECT_NAME}) 

# install launch
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
