/* ----------------------------------------------------------------------------
 * Copyright 2023, Speike <shao-haoluo@foxmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file       4.4.2_trajectoryError.cpp
 * @author     Speike
 * @date       2023/06/02 12:06:23
 * @brief      读取estimated.txt和groundtruth.txt,使用ros2在rviz2中绘制,并计算
 *             绝对轨迹误差ATE_all 绝对平移误差ATE_tran
 *             由于相对轨迹误差还需要考虑时间间隔,这里没有进行计算
**/


#include <fstream>
#include <filesystem>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sophus/se3.hpp"
using namespace std;
using namespace std::chrono_literals;

string estipath = "./src/ch4_lie_theory/estimated.txt";
string gtpath = "./src/ch4_lie_theory/groundtruth.txt";
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;


pair<double,double> CalculateError(TrajectoryType &esti_traj_, TrajectoryType &gt_traj_)
{
    double ATE_all=0, ATE_tran=0;
    double N = esti_traj_.size();
    for (size_t i = 0; i < esti_traj_.size(); i++)
    {
        ATE_all += pow((gt_traj_[i].inverse() * esti_traj_[i]).log().norm(),2);
        ATE_tran += pow((gt_traj_[i].inverse() * esti_traj_[i]).translation().norm(), 2);
    }
    ATE_all = sqrt(ATE_all / N);    //绝对轨迹误差ATE_all
    ATE_tran = sqrt(ATE_tran / N);  //绝对平移误差ATE_tran
    return make_pair(ATE_all, ATE_tran);
}

class TrajectoryPublisher : public rclcpp::Node
{
public:
    TrajectoryPublisher() : Node("traj_pub_node")
    {
        RCLCPP_INFO(this->get_logger(), "轨迹发布节点创建");
        infile_esti.open(estipath);
        infile_gt.open(gtpath);
        if(!infile_esti.is_open() || !infile_gt.is_open())
            RCLCPP_ERROR_STREAM(this->get_logger(), "轨迹文件加载出错!!!,当前路径为:" << filesystem::current_path());
        esti_publisher_ = this->create_publisher<nav_msgs::msg::Path>("esti_traj_pub", 10);
        gt_publisher_ = this->create_publisher<nav_msgs::msg::Path>("gt_traj_pub", 10);
        timer_ = this->create_wall_timer(50ms, bind(&TrajectoryPublisher::timer_callback, this));
    }

private:
    void timer_callback(){
        if(!infile_esti.eof()){
            infile_esti >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            esti_traj_msg.header.frame_id = "world";
            esti_traj_msg.header.stamp = rclcpp::Time(time);

            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = esti_traj_msg.header;
            pose_stamped.pose.position.x = tx*5;    // 为了便于查看，放大五倍
            pose_stamped.pose.position.y = ty*5;
            pose_stamped.pose.position.z = tz*5;
            pose_stamped.pose.orientation.x = qx;
            pose_stamped.pose.orientation.y = qy;
            pose_stamped.pose.orientation.z = qz;
            pose_stamped.pose.orientation.w = qw;
            esti_traj_msg.poses.push_back(pose_stamped);
            esti_publisher_->publish(esti_traj_msg);
            // 保存SE(3)
            esti_sophus_vector.push_back(Sophus::SE3d(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz)));
        }
        if(!infile_gt.eof()){
            infile_gt >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            gt_traj_msg.header.frame_id = "world";
            gt_traj_msg.header.stamp = rclcpp::Time(time);

            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = gt_traj_msg.header;
            pose_stamped.pose.position.x = tx*5;    // 为了便于查看，放大五倍
            pose_stamped.pose.position.y = ty*5;
            pose_stamped.pose.position.z = tz*5;
            pose_stamped.pose.orientation.x = qx;
            pose_stamped.pose.orientation.y = qy;
            pose_stamped.pose.orientation.z = qz;
            pose_stamped.pose.orientation.w = qw;
            gt_traj_msg.poses.push_back(pose_stamped);
            gt_publisher_->publish(gt_traj_msg);
            // 保存SE(3)
            gt_sophus_vector.push_back(Sophus::SE3d(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz)));
        }
        // 轨迹发布结束,计算误差
        if(infile_esti.eof() && infile_gt.eof()) 
        {
            auto [ATE_all, ATE_tran] = CalculateError(esti_sophus_vector, gt_sophus_vector);
            RCLCPP_INFO_STREAM(this->get_logger(), "\n==> 绝对轨迹误差为:" << ATE_all << "\n==> 绝对平移误差为:" << ATE_tran);
            rclcpp::shutdown();
        }
    }
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr esti_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    ifstream infile_esti, infile_gt;
    double time, tx, ty, tz, qx, qy, qz, qw;
    nav_msgs::msg::Path esti_traj_msg, gt_traj_msg;
    TrajectoryType esti_sophus_vector, gt_sophus_vector;    // 储存每帧轨迹的SE(3)
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<TrajectoryPublisher>());
    rclcpp::shutdown();
    return 0;
}
