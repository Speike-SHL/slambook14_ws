cmake_minimum_required(VERSION 3.16)
project(ch6_nonlinear_optimization)

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

set(CMAKE_BUILD_TYPE Debug)   # 若要对比算法速度，最好使用Release模式, 否则会很慢

# 添加module路径
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/module)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)
find_package(glog REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(include)

# 编译设置
add_executable(6.3.1_byhand src/6.3.1_byhand.cpp)
target_link_libraries(6.3.1_byhand ${OpenCV_LIBS})
add_executable(6.3.2_ceresCurveFitting src/6.3.2_ceresCurveFitting.cpp)
target_link_libraries(6.3.2_ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
add_executable(6.3.3_g2oCurveFitting src/6.3.3_g2oCurveFitting.cpp)
target_link_libraries(6.3.3_g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} glog::glog)

# install settings
install(TARGETS
  6.3.1_byhand
  6.3.2_ceresCurveFitting
  6.3.3_g2oCurveFitting
  DESTINATION lib/${PROJECT_NAME})

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
