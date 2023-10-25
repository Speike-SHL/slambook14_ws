cmake_minimum_required(VERSION 3.5)
project(ch10_back_end_2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Release)   # 若要对比算法速度，最好使用Release模式, 否则会很慢
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

# 添加module路径
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/module)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(G2O REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${G2O_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIRS})

SET(G2O_LIBS g2o_core g2o_stuff g2o_types_slam3d)

# 编译设置
add_executable(10.3.1_pose_graph_g2o_SE3 src/10.3.1_pose_graph_g2o_SE3.cpp)
target_link_libraries(10.3.1_pose_graph_g2o_SE3 ${G2O_LIBS} -lstdc++fs glog::glog)
add_executable(10.3.2_pose_graph_g2o_lie_algebra src/10.3.2_pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(10.3.2_pose_graph_g2o_lie_algebra ${G2O_LIBS} -lstdc++fs glog::glog)


# install settings
install(TARGETS
  10.3.1_pose_graph_g2o_SE3
  10.3.2_pose_graph_g2o_lie_algebra
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
