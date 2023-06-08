/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       5.4.1_stereoVision_node.cpp
 * @author     Speike
 * @date       2023/06/07 20:30:56
 * @brief      5.4.1双目视觉, 由左右双目图像计算视察图, 然后在rviz2中绘制点云图
 *             sensor_msgs/pointcloud2的使用
**/

#include <filesystem>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono_literals;

string left_img_path = "./src/ch5_camera_model/data/5.4.1_left.png";
string right_img_path = "./src/ch5_camera_model/data/5.4.1_right.png";
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double b = 0.573;

class Pointcloud2Publisher : public rclcpp::Node
{
public:
    Pointcloud2Publisher() : Node("pointcloud2_pub_node")
    {
        /*====================================== ros2相关 ======================================*/
        pointcloud2_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud2_pub", 100);
        timer_ = this->create_wall_timer(500ms, bind(&Pointcloud2Publisher::timer_callback, this));
        /*====================================== 读取图像 ======================================*/
        cv::Mat left_img = cv::imread(left_img_path, CV_8UC1);
        cv::Mat right_img = cv::imread(right_img_path, CV_8UC1);
        if(left_img.data == nullptr || right_img.data == nullptr)
        {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "读取图像" << left_img_path << "和" 
                                << right_img_path << "出错, 当前路径为" 
                                << filesystem::current_path() << endl);
        }
        /*========== 使用opencv函数计算像素平面的视差,注意课本公式5.15中d为成像平面的视差 ==========*/
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            0, 16 * 6, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32); // 不用知道参数怎么来的
        cv::Mat disparity_sgbm, disparity;
        sgbm->compute(left_img, right_img, disparity_sgbm); //disparity_sgbm内为CV_16S的数据,
        disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); // 要转为32位浮点数,最后的视差图
        // cv::imshow("disparity", disparity/96.0);    // 显示时候进行一下归一化便于观察
        // cv::waitKey(0);
        /*============================ 准备sensor_msgs/PointCloud2 ============================*/
        pointcloud2_msg.header.frame_id = "left_camera";
        pointcloud2_msg.header.stamp = this->now();
        // 也可将height设为1代表点云是无序的,此时width可不设置默认等于点云长度
        pointcloud2_msg.height = left_img.rows; 
        pointcloud2_msg.width = left_img.cols;
        pointcloud2_msg.is_bigendian = false; // 小端存储,高字节为0
        pointcloud2_msg.is_dense = false;   // 点云中包含无效点
        sensor_msgs::PointCloud2Modifier modifier(pointcloud2_msg);
        modifier.setPointCloud2Fields(4,//n_fields,name,count,datatype
                                    "x", 1, sensor_msgs::msg::PointField::FLOAT32,    
                                    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                    "rgb", 1, sensor_msgs::msg::PointField::UINT32);//4*8=32分别每8位代表一个字母,rgb顺序为*rgb,rgba顺序为argb
        // modifier.resize(height_ * width_);   当只设置了height或width其中之一时可加这句，也可不加
        sensor_msgs::PointCloud2Iterator<float> iter_x(pointcloud2_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(pointcloud2_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(pointcloud2_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(pointcloud2_msg, "rgb");
        /*=============================== 开始恢复深度, 生成点云 ===============================*/
        for (int v = 0; v < left_img.rows; v++)
            for (int u = 0; u < left_img.cols; u++, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb)
            {
                if(disparity.ptr<float>(v)[u] <= 0.0 || disparity.ptr<float>(v)[u]>=96.0)
                    continue;
                // 计算归一化坐标
                double x = (u - cx) / fx, y = (v - cy) / fy;
                // 由视差计算深度,课本公式5.15
                double depth = fx * b / disparity.ptr<float>(v)[u];
                // 存入pointcloud2中
                *iter_x = x * depth;
                *iter_y = y * depth;
                *iter_z = depth;
                uint8_t r = left_img.ptr<uchar>(v)[u];
                uint8_t g = left_img.ptr<uchar>(v)[u];
                uint8_t b = left_img.ptr<uchar>(v)[u];
                uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                *iter_rgb = rgb;
            }
    }

private:
    void timer_callback()
    {
        pointcloud2_publisher_->publish(pointcloud2_msg);
    }
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    sensor_msgs::msg::PointCloud2 pointcloud2_msg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<Pointcloud2Publisher>());
    rclcpp::shutdown();
    return 0;
}
