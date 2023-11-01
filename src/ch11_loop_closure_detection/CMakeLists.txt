cmake_minimum_required(VERSION 3.5)
project(ch11_loop_closure_detection)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Debug)   # 若要对比算法速度，最好使用Release模式, 否则会很慢
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)

include_directories(${DBoW3_INCLUDE_DIRS})

# 编译设置
add_executable(11.3.2_DBoW3_training_dictionary src/11.3.2_DBoW3_training_dictionary.cpp)
target_link_libraries(11.3.2_DBoW3_training_dictionary ${OpenCV_LIBS} ${DBoW3_LIBS} -lstdc++fs)
add_executable(11.4.2_DBoW3_similarity_calculation src/11.4.2_DBoW3_similarity_calculation.cpp)
target_link_libraries(11.4.2_DBoW3_similarity_calculation ${OpenCV_LIBS} ${DBoW3_LIBS} -lstdc++fs)

# install settings
install(TARGETS
  11.3.2_DBoW3_training_dictionary
  11.4.2_DBoW3_similarity_calculation
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
