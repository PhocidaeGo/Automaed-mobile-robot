#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <algorithm> // For sorting
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Boolean_set_operations_2.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <filesystem>

struct Waypoint {
    float x, y, z;
};

// RANSAC parameters
const double ignore_threshold = 0.01; // The value determines how many points of the whole cloud will be ignored
const double distance_threshold = 0.02;

// CV parameters
const int image_size = 640; //YOLOv5 default input size
const float z_coordinate = 0.1;
const float area_threshold = 1000.0;
const float scale_down = 0.9;
const float scale_up = 1.3;


// Helper function to preprocess the binary image for YOLOv5
cv::Mat preprocessImage(const cv::Mat& binary_image, int input_size) {
    cv::Mat resized_image, float_image;
    cv::resize(binary_image, resized_image, cv::Size(input_size, input_size));
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0); // Normalize to [0,1]
    return float_image;
}

// Helper function to postprocess the model's output
std::vector<std::vector<cv::Point>> postprocessResults(const std::vector<float>& outputs, float conf_threshold, int input_size) {
    std::vector<std::vector<cv::Point>> rectangles;
    const int num_classes = 1; // Assume detecting rectangles only
    const int num_attrs = 5 + num_classes; // x, y, w, h, conf + num_classes
    const int grid_size = static_cast<int>(std::sqrt(outputs.size() / num_attrs));
    
    for (int i = 0; i < outputs.size() / num_attrs; ++i) {
        float conf = outputs[i * num_attrs + 4];
        if (conf < conf_threshold) continue;

        float cx = outputs[i * num_attrs] * input_size;
        float cy = outputs[i * num_attrs + 1] * input_size;
        float w = outputs[i * num_attrs + 2] * input_size;
        float h = outputs[i * num_attrs + 3] * input_size;

        cv::Point tl(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2));
        cv::Point br(static_cast<int>(cx + w / 2), static_cast<int>(cy + h / 2));
        rectangles.push_back({tl, cv::Point(tl.x, br.y), br, cv::Point(br.x, tl.y)});
    }
    return rectangles;
}

// Main detection function
std::vector<std::vector<cv::Point>> detectShapes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Convert point cloud to a 2D binary image
    cv::Mat binary_image = cv::Mat::zeros(image_size, image_size, CV_8UC1);
    for (const auto& point : cloud->points) {
        int x = static_cast<int>((point.x + 10) * (image_size / 20.0));
        int y = static_cast<int>((point.y + 10) * (image_size / 20.0));
        if (x >= 0 && x < image_size && y >= 0 && y < image_size) {
            binary_image.at<uchar>(y, x) = 255;
        }
    }

    // Preprocess binary image for YOLOv5
    cv::Mat input_image = preprocessImage(binary_image, image_size);

    // Load ONNX Runtime session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ShapeDetection");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, "/home/yuanyan/autonomous-exploration-with-lio-sam/yolov5m.onnx", session_options);

    // Get input and output info
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();

    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();

    // Prepare input tensor
    std::vector<int64_t> input_tensor_shape = {1, 3, image_size, image_size};
    std::vector<float> input_tensor_values(input_image.total() * 3, 0.0f);

    // Expand single channel to 3 channels (C, H, W)
    for (int i = 0; i < input_image.total(); ++i) {
        float value = input_image.data[i] / 255.0f; // Normalize to [0, 1]
        input_tensor_values[i] = value;             // Red channel
        input_tensor_values[i + input_image.total()] = value; // Green channel
        input_tensor_values[i + 2 * input_image.total()] = value; // Blue channel
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    // Run inference
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // Extract output data
    auto* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> outputs(output_data, output_data + output_size);

    // Postprocess output to extract rectangles
    std::vector<std::vector<cv::Point>> detected_rectangles = postprocessResults(outputs, 0.25, image_size);

    // Visualize results
    cv::Mat result_image;
    cv::cvtColor(binary_image, result_image, cv::COLOR_GRAY2BGR);
    for (const auto& rect : detected_rectangles) {
        cv::polylines(result_image, rect, true, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Binary Image", binary_image);
    cv::imshow("Detected Rectangles", result_image);
    cv::waitKey(0);

    return detected_rectangles;
}


std::vector<cv::Point2f> transformToRealWorld(const std::vector<cv::Point>& rectangle, int image_size, float map_range = 20.0) {
    std::vector<cv::Point2f> real_world_coords;
    for (const auto& vertex : rectangle) {
        float real_x = (vertex.x * (map_range / image_size)) - 10; // Inverse of mapping formula
        float real_y = (vertex.y * (map_range / image_size)) - 10;
        real_world_coords.emplace_back(real_x, real_y);
    }
    return real_world_coords;
}


// Generate waypoints for a detected rectangles
std::vector<Waypoint> generateWaypointsFromRectangles(const std::vector<std::vector<cv::Point>>& rectangles, float zCoordinate) {

    std::vector<Waypoint> all_waypoints;

    // Find the largest rectangle (by area) to use as the bounding box
    const std::vector<cv::Point>* largest_rectangle = nullptr;
    double max_area = 0.0;
    for (const auto& rectangle : rectangles) {
        double area = cv::contourArea(rectangle);
        if (area > max_area) {
            max_area = area;
            largest_rectangle = &rectangle;
        }
    }

    if (!largest_rectangle) {
        // No rectangles found
        return all_waypoints;
    }

    // Transform bounding box to real-world coordinates
    auto bounding_box_real = transformToRealWorld(*largest_rectangle, image_size);

    // Check if a point is inside the bounding box
    auto isInsideBoundingBox = [&](const cv::Point2f& point) -> bool {
        std::vector<cv::Point2f> polygon(bounding_box_real.begin(), bounding_box_real.end());
        return cv::pointPolygonTest(polygon, point, false) >= 0; // Returns true if point is inside
    };

    // Generate waypoints for all rectangles
    for (const auto& rectangle : rectangles) {
        bool is_bounding_box = (&rectangle == largest_rectangle);

        std::vector<Waypoint> waypoints;
        // Scale rectangle for waypoint generation
        float scale_factor = is_bounding_box ? scale_down : scale_up;
        cv::Point2f center(0, 0);

        // Transform rectangle vertices back to real-world coordinates
        auto real_world_rectangle = transformToRealWorld(rectangle, image_size);

        for (const auto& vertex : real_world_rectangle) {
            center += cv::Point2f(vertex.x, vertex.y); // Accumulate vertex positions
        }
        center *= (1.0 / real_world_rectangle.size()); // Compute the center

        for (const auto& vertex : real_world_rectangle) {
            cv::Point2f vertex_f(vertex.x, vertex.y); // Convert to float point
            cv::Point2f scaled_vertex = center + (vertex_f - center) * scale_factor;

            // Check if the scaled vertex is inside the bounding box
            if (isInsideBoundingBox(scaled_vertex)) {
                waypoints.push_back({scaled_vertex.x, scaled_vertex.y, z_coordinate});
            }
        }

        // Add waypoints to the global list
        all_waypoints.insert(all_waypoints.end(), waypoints.begin(), waypoints.end());
    }
    return all_waypoints;
}

// Visualize waypoints in the PCL visualizer
void visualizeWaypoints(pcl::visualization::PCLVisualizer::Ptr &viewer, const std::vector<Waypoint> &waypoints) {
    int waypoint_id = 0;
    for (const auto &wp : waypoints) {
        std::string point_id = "waypoint_" + std::to_string(waypoint_id);
        viewer->addSphere<pcl::PointXYZ>(pcl::PointXYZ(wp.x, wp.y, wp.z), 0.1, 0.0, 0.0, 1.0, point_id);
        waypoint_id++;
    }
}


class LidarMapProcessor : public rclcpp::Node {
public:
    LidarMapProcessor() : Node("lidar_map_processor") {
        RCLCPP_INFO(this->get_logger(), "Starting Lidar Map Processor Node...");
        processPointCloud();
    }
    

private:    
    void processPointCloud() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr wall_features(new pcl::PointCloud<pcl::PointXYZ>());

        // Load the PCD file
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        std::string file_path = "/home/yuanyan/Downloads/LOAM_Backup/cloudGlobal.pcd";
        //Read the .pcd file and store into the cloud object (of type pcl::PointCloud<pcl::PointXYZ>).
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", file_path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu points from %s", cloud->size(), file_path.c_str());

        // Save the processed file as a .ply
        std::string output_file = "/home/yuanyan/autonomous-exploration-with-lio-sam/src/vehicle_simulator/mesh/test/preview/pointcloud.ply";
        if (pcl::io::savePLYFile(output_file, *cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't write file: %s", output_file.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Processed cloud saved as: %s", output_file.c_str());
        }

        // Prepare visualizer
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Extracted Planes"));
        viewer->setBackgroundColor(0, 0, 0);
        //viewer->addPointCloud<pcl::PointXYZ>(cloud, "original_cloud");
        //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");

        // Plane model segmentation for feature exraction of walls and ceiling. Ref: https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html
        // RANSAC method
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold); // Adjust as needed

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        int i = 0;
        
        // Ensures the segmentation loop does not continue indefinitely. If the remaining cloud becomes too small (less than 30% of the original cloud), the loop stops.
        while (cloud->size() > ignore_threshold * cloud_filtered->size()) {
            // Segment the largest planar component of rest of cloed
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);//*inliers: A list of indices indicating which points in the cloud belong to the detected plane.
                                                    //*coefficients: A set of model coefficients (e.g., ax + by + cz + d = 0) describing the detected plane.
            if (inliers->indices.empty()) {
                RCLCPP_INFO(this->get_logger(), "No more planar components found.");
                break;
            }

            RCLCPP_INFO(this->get_logger(), "Plane %d: %lu points", i++, inliers->indices.size());

            // Extract the plane
            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>()); //Creates a new point cloud (plane_cloud) to store the points belonging to the detected plane.
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false); // Extract only points in inliers.
            extract.filter(*plane_cloud);

            // Determine if the plane is a wall or ceiling
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*plane_cloud, centroid);

            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];
            float d = coefficients->values[3];
            Eigen::Vector3f normal(a, b, c);

            std::string plane_name = "Plane_" + std::to_string(i);
            if (std::abs(normal.dot(Eigen::Vector3f(0, 0, 1))) < 0.2) {

                RCLCPP_INFO(this->get_logger(), "Detected vertical wall at centroid (%.2f, %.2f, %.2f)",
                            centroid[0], centroid[1], centroid[2]);

                // Save the wall point cloud to wall_features
                *wall_features += *plane_cloud;

                viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                            1.0, 0.0, 0.0, plane_name); //Showed in red

            } else if (normal.dot(Eigen::Vector3f(0, 0, -1)) > 0.9) {
                RCLCPP_INFO(this->get_logger(), "Detected horizontal ceiling at centroid (%.2f, %.2f, %.2f)",
                            centroid[0], centroid[1], centroid[2]);
                viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                            0.0, 1.0, 0.0, plane_name); //Showed in green

            } else {
                RCLCPP_INFO(this->get_logger(), "Other plane detected.");
            }

            // Remove the extracted plane from the cloud
            extract.setNegative(true);
            extract.filter(*cloud_filtered);
            cloud.swap(cloud_filtered);
            i++;
            }

        std::vector<Waypoint> waypoints; // List to store all waypoints
        // Detect rectangles
        auto rectangles = detectShapes(wall_features);
        // Generate waypoints
        std::vector<Waypoint> all_waypoints = generateWaypointsFromRectangles(rectangles, z_coordinate);        

        // Visualize waypoints
        visualizeWaypoints(viewer, all_waypoints);

        // Visualize
        viewer->spin();
    }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarMapProcessor>());
  rclcpp::shutdown();
  return 0;
}