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

## 程序目录  
1. ch3_rigid_body_trans(第三讲：三维空间刚体运动)
    - `ros2 run ch3_rigid_body_trans 3.6.1_useEigen`
    - `ros2 run ch3_rigid_body_trans 3.6.2_coordinateTrans`
    - `ros2 launch ch3_rigid_body_trans 3.7.1_plotTrajectory.launch.py`
2. ch4_lie_theory(第四讲：李群与李代数)
    - `ros2 run ch4_lie_theory 4.4.1_useSophus`
    - `ros2 launch ch4_lie_theory 4.4.2_trajectoryError.launch.py`
3. ch5_camera_model(第五讲：相机与图像)
    - `ros2 run ch5_camera_model 5.3.1_opencvBasic`
    - `ros2 run ch5_camera_model 5.3.2_undistortImage`
    - `ros2 launch ch5_camera_model 5.4.1_stereoVision_node.launch.py`
    - `ros2 launch ch5_camera_model 5.4.2_rgbdJointMap_node.launch.py`
4. ch6_nonlinear_optimization(第六讲：非线性优化)
    - `ros2 run ch6_nonlinear_optimization 6.3.1_byhand`
    - `ros2 run ch6_nonlinear_optimization 6.3.2_ceresCurveFitting`
    - `ros2 run ch6_nonlinear_optimization 6.3.3_g2oCurveFitting`
5. ch7_visual_odometry_1(第七讲：视觉里程计1)
    - `ros2 run ch7_visual_odometry_1 7.2.1_orb_opencv` 使用Opencv实现orb特征点提取匹配
    - `ros2 run ch7_visual_odometry_1 7.2.2_orb_byhand` 手写orb特征点的提取与匹配
