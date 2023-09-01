/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       5.4.2_rgbdJointMap_node.cpp
 * @author     Speike
 * @date       2023/06/08 14:15:49
 * @brief      5.4.2 RGBD联合建图, 用data/5.4.2_rgbd/下的数据,
 *             位姿的记录格式为: x y z qx qy qz qw
 *             与双目中不同之处在于, 发布点云数据使用了PCL库到Sensor_msgs的转化
 **/
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

using namespace std;
using namespace std::chrono_literals;

// 相机内参
double cx = 325.5, cy = 253.5, fx = 518.0, fy = 519.0;
// 深度图
double depthScale = 1000.0;
string color_img_path = "./src/ch5_camera_model/data/5.4.2_rgbd/color/";
string depth_img_path = "./src/ch5_camera_model/data/5.4.2_rgbd/depth/";
string pose_path = "./src/ch5_camera_model/data/5.4.2_rgbd/pose.txt";

class Pointcloud2Publisher : public rclcpp::Node
{
public:
    Pointcloud2Publisher(
        const vector<cv::Mat>& colorImgs_, const vector<cv::Mat>& depthImgs_,
        const vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& poses_)
        : Node("pointcloud2_pub_node")
    {
        /*================================= ROS2相关
         * ===============================*/
        pointcloud2_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud2_pub",
                                                                  100);
        timer_ = this->create_wall_timer(
            50ms, std::bind(&Pointcloud2Publisher::timer_callback, this));

        timer_ = this->create_wall_timer(
            50ms, std::bind(&Pointcloud2Publisher::timer_callback, this));

        /*=================== 准备pcl::PointCloud<pcl::PointXYZRGB>
         * ================*/
        cloud_.height = 1;
        cloud_.width = 2000000;
        cloud_.is_dense = true;

        /*================== 读取图像颜色和深度,左乘T_wc全部转化到世界系
         * ===============*/
        for (int i = 0; i < 5; i++)
        {
            RCLCPP_INFO(this->get_logger(), "转换图像中: %d.", i + 1);
            const cv::Mat& colorImg = colorImgs_[i];
            const cv::Mat& depthImg = depthImgs_[i];
            const Sophus::SE3d T_wc = poses_[i];

            for (int v = 0; v < colorImg.rows; v++)
            {
                for (int u = 0; u < colorImg.cols; u++)
                {
                    uint16_t depth = depthImg.ptr<uint16_t>(v)[u];
                    if (depth == 0) continue;  // 表示没有测量到

                    if (depth == 0) continue;  // 表示没有测量到

                    // 归一化坐标
                    double x = (u - cx) / fx, y = (v - cy) / fy;
                    // 三维坐标
                    double z = double(depth) / depthScale;
                    x = x * z;
                    y = y * z;
                    // 转化到世界系
                    Sophus::Vector3d point_W = T_wc * Sophus::Vector3d(x, y, z);
                    // 封装点云数据
                    pcl::PointXYZRGB point;
                    point.x = float(point_W[0]);
                    point.y = float(point_W[2]);  // 换一下顺序,便于rviz中看
                    point.z = -float(point_W[1]);
                    point.b = colorImg.ptr<uchar>(v)[3 * u];
                    point.g = colorImg.ptr<uchar>(v)[3 * u + 1];
                    point.r = colorImg.ptr<uchar>(v)[3 * u + 2];
                    cloud_.push_back(point);
                }
            }
        }

        cout << "五张图像总点数为: " << cloud_.size() << endl;

        /*============================== 进行体素滤波 ============================*/
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>(cloud_));
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_ptr);
        sor.setLeafSize(0.015f, 0.015f, 0.015f);
        sor.filter(*cloud_ptr);
        cloud_ = *cloud_ptr;
        cout << "进行体素滤波后的点数为:" << cloud_.size() << endl;

        /*========================== 使用pcl进行可视化 ============================*/
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
            cloud_ptr);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ptr, rgb, "cloud");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

        /*========================== 转化pcl为sensor_msgs ========================*/
        pcl::toROSMsg(cloud_, pointcloud2_msg);  // 一定先转化再加header头
        pointcloud2_msg.header.frame_id = "world";
        pointcloud2_msg.is_bigendian = false;

        cout << "构造函数结束" << endl;
    }

private:
    void timer_callback()
    {
        if (!viewer->wasStopped())
            viewer->spinOnce(100);
        else
        {
            cout << "用户关闭了PCL Viewer窗口并杀死了节点" << endl;
            rclcpp::shutdown();
        }
        /*========================== 发布点云话题 ===============================*/
        /*========================== 发布点云话题 ===============================*/
        pointcloud2_msg.header.stamp = this->now();
        pointcloud2_publisher_->publish(pointcloud2_msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pointcloud2_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    sensor_msgs::msg::PointCloud2 pointcloud2_msg;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_;
    pcl::visualization::PCLVisualizer::Ptr viewer =
        pcl::make_shared<pcl::visualization::PCLVisualizer>("Point Cloud Viewer");
};

int main(int argc, char** argv)
{
    // 首先处理图像数据文件
    vector<cv::Mat> colorImgs, depthImgs;
    for (int i = 1; i < 6; i++)
    {
        string path_color = color_img_path + to_string(i) + ".png";
        string path_depth = depth_img_path + to_string(i) + ".pgm";
        cv::Mat temp_color = cv::imread(path_color, 1);
        cv::Mat temp_depth = cv::imread(path_depth, -1);
        if (temp_color.data == nullptr || temp_depth.data == nullptr)
        {
            cerr << "读取图像" << path_color << "或" << path_depth
                 << "出错, 当前路径为" << filesystem::current_path() << endl;
            return EXIT_FAILURE;
        }
        colorImgs.push_back(temp_color);
        depthImgs.push_back(temp_depth);
    }
    // 然后处理姿态数据文件
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses;
    ifstream infile(pose_path);
    if (!infile)
    {
        cerr << "读取姿态文件:" << pose_path << "出错, 当前路径为"
             << filesystem::current_path() << endl;
        return EXIT_FAILURE;
    }
    double tx, ty, tz, qx, qy, qz, qw;
    for (int i = 0; i < 5; i++)
    {
        infile >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        poses.push_back(Sophus::SE3d(Eigen::Quaterniond(qw, qx, qy, qz),
                                     Sophus::Vector3d(tx, ty, tz)));
    }
    // 创建ROS2节点,传入数据
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<Pointcloud2Publisher>(colorImgs, depthImgs, poses));
    rclcpp::shutdown();
}
