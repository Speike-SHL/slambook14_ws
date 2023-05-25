cmake_minimum_required(VERSION 3.5)
project(ch3_rigid_body_trans)

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
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

# 编译设置
add_executable(3.6.1_useEigen src/3.6.1_useEigen.cpp)
add_executable(3.6.2_coordinateTrans src/3.6.2_coordinateTrans.cpp)
add_executable(3.7.1_plotTrajectory_node src/3.7.1_plotTrajectory_node.cpp)
ament_target_dependencies(3.7.1_plotTrajectory_node 
    rclcpp 
    nav_msgs 
    geometry_msgs
    )

# install settings
install(TARGETS 
    3.7.1_plotTrajectory_node
    3.6.1_useEigen
    3.6.2_coordinateTrans 
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
