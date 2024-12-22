#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <geometry_msgs/msg/twist_stamped.hpp>

struct Waypoint {
    float x, y, z;
};

//Read waypoints from a file and publish
std::vector<Waypoint> readWaypointsFromFile(const std::string& filename) {
    std::vector<Waypoint> waypoints;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for reading");
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Waypoint waypoint;
        iss >> waypoint.x >> waypoint.y >> waypoint.z;
        waypoints.push_back(waypoint);
    }
    file.close();
    return waypoints;
}

class WaypointPublisher : public rclcpp::Node {
public:
    WaypointPublisher()
        : Node("waypoint_publisher") {
        publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/way_point", 13);
        publishWaypoints();
    }

private:
    void publishWaypoints() {
        std::vector<Waypoint> all_waypoints = readWaypointsFromFile("waypoints.txt");

        for (const auto &waypoint : all_waypoints) {
            auto message = geometry_msgs::msg::PointStamped();

            // Fill in the header
            message.header.stamp = this->get_clock()->now();
            message.header.frame_id = "map";

            // Fill in the waypoint coordinates
            message.point.x = waypoint.x;
            message.point.y = waypoint.y;
            message.point.z = waypoint.z;

            // Publish the message
            publisher_->publish(message);

            // Sleep for a short duration to allow the system to process the message
            rclcpp::sleep_for(std::chrono::milliseconds(10000));
        }
    }

    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr publisher_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WaypointPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
