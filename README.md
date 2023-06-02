# slambook14_ws
视觉SLAM十四讲学习相关代码，使用ROS2构建，抛弃了Pangolin的使用

src下每章都是一个功能包

在vscode中打开对应的程序文件直接Ctrl+Shift+B就能自动colcon build然后ros2 run 运行(已经在.vscode中的tasks.json中写好了默认build任务)

不管有没有写成ros2的节点，都可以用ros2 run <功能包> <可执行文件>运行

ros2语法：
- 创建功能包: `ros2 pkg create <package_name> --build-type ament_cmake --dependencies rclcpp std_msgs`
