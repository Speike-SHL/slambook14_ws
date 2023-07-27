# slambook14_ws
视觉SLAM十四讲学习相关代码，使用ROS2构建，抛弃了Pangolin的使用

src下每章都是一个功能包

在vscode中打开对应的程序文件直接`Ctrl+Shift+B`就能自动`colcon build`然后`ros2 run`运行(已经在`.vscode`中的`tasks.json`中写好了默认`build`任务),但是注意在`tasks.json`的`ros2 run`任务里要注意使用的终端是`bash`还是`zsh`，修改不同的`source`。

调试:
1. 在CMakeLists中添加`set(CMAKE_BUILD_TYPE Debug)`  
2. 修改`launch.json`中的可执行文件路径中的功能包名。  

不管有没有写成ros2的节点，都可以用`ros2 run <功能包> <可执行文件>`运行

ros2语法：
- 创建功能包: `ros2 pkg create <package_name> --build-type ament_cmake --dependencies rclcpp std_msgs`
