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
    - `ros2 run ch7_visual_odometry_1 7.4_polar_constriant_2d2d` 已知匹配点用对极几何求相邻两帧间的运动R、t，求基础矩阵F、本质矩阵E和单应矩阵H
    - `ros2 run ch7_visual_odometry_1 7.6.1_triangulation` 三角化，使用opencv提供的函数进行三角化
    - `ros2 run ch7_visual_odometry_1 7.8_PnP_3d2d` 分别使用OpenCV的EPnP, 手写高斯牛顿法解BAPnP, 使用G2O求解BAPnP 以及使用Ceres-solver解BAPnP
    - `ros2 run ch7_visual_odometry_1 7.10_ICP_3d3d` 分别使用SVD法、G2O上的非线性优化、CERES上的非线性优化等方法求解ICP问题
6. ch8_visual_odometry_2(第八讲：视觉里程计2)
    - `ros2 run ch8_visual_odometry_2 8.3_LK_optical_flow` OpenCV实现光流法、手写高斯牛顿实现正向光流, 光流金字塔实现反向光流, 同时调用tbb中的parallel_for_并行的计算每个关键点的光流估计
    - `ros2 run ch8_visual_odometry_2 8.5_direct_method` OpenCV没有直接支持直接法, 使用单层和多层直接法。由于点过多, 误差过大，优化时没法累加矩阵GN，因此使用SUM(H)deltax = SUM(b)。但是这一节直接法的效果感觉并不好,从图上就能看出来。
7. ch9_back_end_1(第九讲：后端1)
    - `ros2 run ch9_back_end_1 9.3_ceres_BA` 大型BA与图优化 , 已知 相机外参R_cw,t_cw(或相机在世界系下的位姿R_wc t_wc) 、三维点位置、对应相机的像素坐标 , **同时优化**世界系下的相机位姿、三维点位置、与相机内参f、k1、k2
    - `ros2 run ch9_back_end_1 9.4_g2o_BA` 大型BA与图优化 , 已知 相机外参R_cw,t_cw(或相机在世界系下的位姿R_wc t_wc) 、三维点位置、对应相机的像素坐标 , **同时优化**世界系下的相机位姿、三维点位置、与相机内参f、k1、k2。与ceres不同的是, 这里需要手动设置边缘化, 而且需要把优化结果放回对应内存。这个文件实现的g2o没有给定雅可比矩阵。比ceres运行快。
