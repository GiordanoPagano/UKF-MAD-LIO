#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

struct TumPose{
    double timestamp;
    double tx;
    double ty;
    double tz;
    double qx;
    double qy;
    double qz;
    double qw;
};

std::vector<TumPose> loadTumFile(const std::string& filename){
    std::vector<TumPose> poses;
    std::ifstream file(filename);
    TumPose p;

    while(file >> p.timestamp >> p.tx >> p.ty >> p.tz >> p.qx >> p.qy >> p.qz >> p.qw){
        poses.push_back(p);
    }
    return poses;
}

nav_msgs::msg::Path buildPath(const std::vector<TumPose>& poses, const std::string& frame){
    nav_msgs::msg::Path path;
    path.header.frame_id = frame;

    for(const auto& p : poses){
        geometry_msgs::msg::PoseStamped pose;

        pose.header.frame_id = frame;
        pose.pose.position.x = p.tx;
        pose.pose.position.y = p.ty;
        pose.pose.position.z = p.tz;
        pose.pose.orientation.x = p.qx;
        pose.pose.orientation.y = p.qy;
        pose.pose.orientation.z = p.qz;
        pose.pose.orientation.w = p.qw;

        path.poses.push_back(pose);
    }
    return path;
}

class TrajectoryVisualizer : public rclcpp::Node {
public:
    TrajectoryVisualizer() : Node("trajectory_visualizer"){
        gt_pub      = create_publisher<nav_msgs::msg::Path>("/gt_path", 1);
        fastlio_pub = create_publisher<nav_msgs::msg::Path>("/fastlio_path", 1);
        ukf_pub     = create_publisher<nav_msgs::msg::Path>("/ukf_path", 1);

        std::string gt_file      = "/home/giordano/Desktop/UKF_ICP/gt_tum.txt";
        std::string fastlio_file = "/home/giordano/Desktop/UKF_ICP/new_FAST-LIO_tum.txt";
        std::string ukf_file     = "/home/giordano/Desktop/UKF_ICP/new_est_lio_und_s_0.txt";

        auto gt      = loadTumFile(gt_file);
        auto fastlio = loadTumFile(fastlio_file);
        auto ukf     = loadTumFile(ukf_file);

        gt_path      = buildPath(gt, "map");
        fastlio_path = buildPath(fastlio, "map");
        ukf_path     = buildPath(ukf, "map");

        timer = create_wall_timer(
            std::chrono::milliseconds(500),
            [this](){
                gt_pub->publish(gt_path);
                fastlio_pub->publish(fastlio_path);
                ukf_pub->publish(ukf_path);
            });
    }

private:
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr fastlio_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ukf_pub;

    nav_msgs::msg::Path gt_path;
    nav_msgs::msg::Path fastlio_path;
    nav_msgs::msg::Path ukf_path;

    rclcpp::TimerBase::SharedPtr timer;
};

int main(int argc, char** argv){
    rclcpp::init(argc, argv);

    auto node = std::make_shared<TrajectoryVisualizer>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}
