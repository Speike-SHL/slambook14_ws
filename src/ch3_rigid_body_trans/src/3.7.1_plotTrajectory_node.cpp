#include <fstream>
#include <filesystem>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
using namespace std;
using namespace std::chrono_literals;

string filepath = "./src/ch3_rigid_body_trans/trajectory.txt";

class TrajectoryPublisher : public rclcpp::Node
{
public:
    TrajectoryPublisher() : Node("traj_pub_node")
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "轨迹发布节点创建");
        infile_traj.open(filepath);     // 打开路径文件
        if(!infile_traj.is_open())
            RCLCPP_ERROR_STREAM(this->get_logger(), "轨迹文件加载出错!!!,当前路径为:" << filesystem::current_path());
        publisher_ = this->create_publisher<nav_msgs::msg::Path>("tarj_pub", 10);
        timer_ = this->create_wall_timer(50ms, bind(&TrajectoryPublisher::timer_callback, this));
    }

private:
    void timer_callback(){
        if(!infile_traj.eof()){
            infile_traj >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            tarj_msg.header.frame_id = "world";
            tarj_msg.header.stamp = rclcpp::Time(time);

            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = tarj_msg.header;
            pose_stamped.pose.position.x = tx*5;    // 为了便于查看，放大五倍
            pose_stamped.pose.position.y = ty*5;
            pose_stamped.pose.position.z = tz*5;
            pose_stamped.pose.orientation.x = qx;
            pose_stamped.pose.orientation.y = qy;
            pose_stamped.pose.orientation.z = qz;
            pose_stamped.pose.orientation.w = qw;
            tarj_msg.poses.push_back(pose_stamped);
            publisher_->publish(tarj_msg);
        }
        else
            rclcpp::shutdown();
    }
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    ifstream infile_traj;
    double time, tx, ty, tz, qx, qy, qz, qw;
    nav_msgs::msg::Path tarj_msg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<TrajectoryPublisher>());
    rclcpp::shutdown();
    return 0; 
}
