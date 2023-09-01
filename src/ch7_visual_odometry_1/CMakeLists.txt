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
set(CMAKE_CXX_FLAGS "-O3 ${SSE_FLAGS} -msse4")

# 添加module路径
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/module)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

# 编译设置
add_executable(7.2.1_orb_opencv src/7.2.1_orb_opencv.cpp)
target_link_libraries(7.2.1_orb_opencv ${OpenCV_LIBS} -lstdc++fs)
add_executable(7.2.2_orb_byhand src/7.2.2_orb_byhand.cpp)
target_link_libraries(7.2.2_orb_byhand ${OpenCV_LIBS} -lstdc++fs)
add_executable(7.4_polar_constraint_2d2d src/7.4_polar_constraint_2d2d.cpp)
target_link_libraries(7.4_polar_constraint_2d2d ${OpenCV_LIBS} -lstdc++fs)
add_executable(7.6.1_triangulation src/7.6.1_triangulation.cpp)
target_link_libraries(7.6.1_triangulation ${OpenCV_LIBS} -lstdc++fs)
add_executable(7.8_PnP_3d2d src/7.8_PnP_3d2d.cpp)
target_link_libraries(7.8_PnP_3d2d ${OpenCV_LIBS} ${CERES_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} -lstdc++fs)
add_executable(7.10_ICP_3d3d src/7.10_ICP_3d3d.cpp)
target_link_libraries(7.10_ICP_3d3d ${OpenCV_LIBS} ${CERES_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} -lstdc++fs)

# install settings
install(TARGETS
  7.2.1_orb_opencv
  7.2.2_orb_byhand
  7.4_polar_constraint_2d2d
  7.6.1_triangulation
  7.8_PnP_3d2d
  7.10_ICP_3d3d
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
