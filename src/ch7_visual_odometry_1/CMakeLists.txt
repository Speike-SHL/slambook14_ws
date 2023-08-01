cmake_minimum_required(VERSION 3.5)
project(ch7_visual_odometry_1)

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

set(CMAKE_BUILD_TYPE Release)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

# 编译设置
add_executable(7.2.1_orb_opencv src/7.2.1_orb_opencv.cpp)
target_link_libraries(7.2.1_orb_opencv ${OpenCV_LIBS})
add_executable(7.2.2_orb_byhand src/7.2.2_orb_byhand.cpp)
target_link_libraries(7.2.2_orb_byhand ${OpenCV_LIBS})

# install settings
install(TARGETS
  7.2.1_orb_opencv
  7.2.2_orb_byhand
  DESTINATION lib/${PROJECT_NAME}
)

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
