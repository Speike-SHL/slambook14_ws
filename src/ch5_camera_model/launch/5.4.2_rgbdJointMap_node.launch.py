import os
from launch import LaunchDescription
from launch_ros.actions import Node
# ------------------------ 封装终端指令相关类 ------------------------
# from launch.actions import ExecuteProcess
# from launch.substitutions import FindExecutable
# ------------------------- 参数声明与获取 ---------------------------
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# -------------------------- 文件包含相关 ----------------------------
# from launch.actions import IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# ---------------------------- 分组相关 ------------------------------
# from launch_ros.actions import PushRosNamespace
# from launch.actions import GroupAction
# ---------------------------- 事件相关 ------------------------------
# from launch.event_handlers import OnProcessStart, OnProcessExit
# from launch.actions import ExecuteProcess, RegisterEventHandler, LogInfo
# --------------------- 获取功能包下share目录路径 ---------------------
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config_file = os.path.join(
        get_package_share_directory('ch5_camera_model'),
        'launch', 
        '5.4.2_show_pointcloud2.rviz'
        )
    print(rviz_config_file)
    t1 = Node(
        package="ch5_camera_model",
        executable="5.4.2_rgbdJointMap_node",
        name="pointcloud2_pub_node",
        output="screen"
        )
    t2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=['-d',rviz_config_file]
        )
    return LaunchDescription([t1,t2])
