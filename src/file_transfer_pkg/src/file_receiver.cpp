#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <fstream>

class FileReceiver : public rclcpp::Node
{
public:
    FileReceiver() : Node("file_receiver")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "file_data", 10,
            std::bind(&FileReceiver::file_callback, this, std::placeholders::_1));
    }

private:
    void file_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        std::ofstream file("/home/yuanyan/Downloads/cloudGlobal.pcd");
        if (!file.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file.");
            return;
        }
        file << msg->data;
        file.close();

        RCLCPP_INFO(this->get_logger(), "File received and saved.");
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FileReceiver>());
    rclcpp::shutdown();
    return 0;
}
