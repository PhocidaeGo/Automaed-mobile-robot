#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <fstream>
#include <sstream>

class FileSender : public rclcpp::Node
{
public:
    FileSender() : Node("file_sender")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("file_data", 10);
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&FileSender::publish_file, this));
    }

private:
    void publish_file()
    {
        std::ifstream file("/home/yuanyan/Downloads/LOAM_Backup/cloudGlobal.pcd");
        if (!file.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file.");
            return;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        auto message = std_msgs::msg::String();
        message.data = buffer.str();
        publisher_->publish(message);

        RCLCPP_INFO(this->get_logger(), "File data published.");
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FileSender>());
    rclcpp::shutdown();
    return 0;
}
