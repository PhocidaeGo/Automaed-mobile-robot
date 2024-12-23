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
#include <thread> // For delay in visualization
#include <chrono> // For std::chrono::seconds
#include <opencv2/opencv.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <fstream>
#include <sstream>

struct Waypoint {
    float x, y, z;
};

// Input point cloud
std::string file_path = "/home/yuanyan/Downloads/LOAM/cloudGlobal.pcd";

// Mode selection
const bool walls_in_room = false; // false means there is no individual walls in the room

// RANSAC parameters
const double ignore_threshold = 0.01; // The value determines how many points of the whole cloud will be ignored.
const double distance_threshold = 0.05; // Too large is also not good

// CV parameters
const int image_size = 500; // Adjust based on the map size, calculate the length each pixel represents in real world. It will also affect the kernel size.
const float z_coordinate = 0.1;
const float image_area_threshold = 300.0; //should be dynamic with image size, can't use area in point cloud, since a random point in space can still be classifie to a wall as long as it's on the same plane.

const float scale_down = 0.9; // Adjust this to control the distance between way points and bounding box (the boundary of the room).
const float scale_up = 1.3; // Adjust this to control the distance between way points and walls in the room.

const double center_distance_threshold = 5; // Filter out duplicate polygons

float min_x, max_x, min_y, max_y;
float width, height, aspect_ratio;

// Way point generator
const double points_min_distance = 1; // meters, filter out duplicate way points


// CGAL typedefs
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef CGAL::Polygon_2<Kernel> Polygon_2;

cv::Mat projectPointCloudToBinaryImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int image_size, float range = 20.0) {
    // Determine the bounding box of the point cloud
    min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::lowest();
    min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::lowest();

    for (const auto& point : cloud->points) {
        if (point.x < min_x) min_x = point.x;
        if (point.x > max_x) max_x = point.x;
        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;
    }

    // Compute the dimensions of the point cloud
    width = max_x - min_x;
    height = max_y - min_y;

    // Compute the aspect ratio and determine image height
    aspect_ratio = height / width;
    int image_height = static_cast<int>(image_size * aspect_ratio);

    // Create a blank binary image
    cv::Mat binary_image = cv::Mat::zeros(image_height, image_size, CV_8UC1);

    // Project each 3D point to 2D and populate the binary image
    for (const auto& point : cloud->points) {
        int x = static_cast<int>((point.x - min_x) * (image_size / width));  // Map to image space (x-axis)
        int y = static_cast<int>((point.y - min_y) * (image_height / height)); // Map to image space (y-axis)
        if (x >= 0 && x < image_size && y >= 0 && y < image_height) {
            binary_image.at<uchar>(y, x) = 255;
        }
    }

    return binary_image;
}

std::vector<cv::Point2f> transformToRealWorld(const std::vector<cv::Point>& rectangle, // Map the 2D image coordinates back to real-world coordinates
                                              int image_size, float map_range = 20.0) {
    int image_height = static_cast<int>(image_size * aspect_ratio);
    
    std::vector<cv::Point2f> real_world_coords;
    for (const auto& vertex : rectangle) {
        float real_x = (vertex.x * (width / image_size)) + min_x; // Inverse of mapping formula
        float real_y = (vertex.y * (height / image_height)) + min_y;
        real_world_coords.emplace_back(real_x, real_y);
    }
    return real_world_coords;
}

cv::Mat enhanceImage(const cv::Mat& dense_image, const cv::Mat& sparse_image) {
    cv::Mat denoise_dense;
    cv::GaussianBlur(dense_image, denoise_dense, cv::Size(3, 3), 0); // better than medianBlur
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)); // Kernel size
    //cv::morphologyEx(dense_image, denoise_dense, cv::MORPH_CLOSE, kernel);
    //cv:imshow("denoise_dense", denoise_dense);

    // Dilate the sparse image to create a region of interest (ROI)
    cv::Mat roi_mask;
    cv::Mat kernel_mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); 
    cv::dilate(sparse_image, roi_mask, kernel_mask);
    //cv::imshow("mask", roi_mask);

    // Create a new blank binary image
    cv::Mat refined_image = cv::Mat::zeros(denoise_dense.size(), CV_8UC1);

    // terate over all pixels in the dense image
    for (int y = 0; y < denoise_dense.rows; ++y) {
        for (int x = 0; x < denoise_dense.cols; ++x) {
            // If the current pixel in the dense image is within the ROI, retain it
            if (roi_mask.at<uchar>(y, x) > 0 && denoise_dense.at<uchar>(y, x) > 125) { //125 for GaussianBlur
                refined_image.at<uchar>(y, x) = 255; // Retain the dense image point
            }
        }
    }
    //cv::imshow("refined_img", refined_image);
    cv::Mat output_image;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)); // Kernel size, 5 for test.world
    cv::morphologyEx(refined_image, output_image, cv::MORPH_CLOSE, kernel);

    // Save the output image
    //cv::imwrite("output_image.png", output_image);
    cv::imshow("output_img", output_image);
    return output_image;
}

std::vector<std::vector<cv::Point>> detectShapes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat dense_image) {
    cv::Mat sparse_image = projectPointCloudToBinaryImage(cloud, image_size);
    //cv::imshow("dense image", dense_image);
    //cv::imshow("sparse image", sparse_image);

    // Enhance the sparse image with the dense image
    cv::Mat binary_image = enhanceImage(dense_image, sparse_image);

    // Detect contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Process each contour using CGAL
    std::vector<std::vector<cv::Point>> polygons;
    cv::Mat result_image = cv::Mat::zeros(binary_image.size(), CV_8UC3);
    cv::cvtColor(binary_image, result_image, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> polygon_centers;     // List to store centers of polygons

    for (const auto& contour : contours) {
        // Simplify the contour using approxPolyDP
        std::vector<cv::Point> simplified_contour;
        double epsilon = 0.03 * cv::arcLength(contour, true); // Adjust epsilon for simplification, 0.02 for test.world
        cv::approxPolyDP(contour, simplified_contour, epsilon, true);

        
        // Convert OpenCV contour to CGAL points
        std::vector<Point_2> contour_points;
        for (const auto& pt : simplified_contour) {
            contour_points.emplace_back(pt.x, pt.y); // Use image coordinates directly
        }

        // Visualize the raw contour
        cv::drawContours(result_image, std::vector<std::vector<cv::Point>>{simplified_contour}, -1, cv::Scalar(255, 255, 255), 1);

        /*
        // Compute convex hull of the points (optional but useful for noisy data)
        std::vector<Point_2> hull;
        CGAL::convex_hull_2(contour_points.begin(), contour_points.end(), std::back_inserter(hull));

        // Draw convex hull
        for (size_t j = 0; j < hull.size(); ++j) {
            int x1 = static_cast<int>(hull[j].x());
            int y1 = static_cast<int>(hull[j].y());
            int x2 = static_cast<int>(hull[(j + 1) % hull.size()].x());
            int y2 = static_cast<int>(hull[(j + 1) % hull.size()].y());
            cv::line(result_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
        }

        // Create a CGAL polygon from the convex hull
        Polygon_2 polygon(hull.begin(), hull.end());
        */

        Polygon_2 polygon(contour_points.begin(), contour_points.end());
        // Check if the polygon is valid
        //if (polygon.size() >= 3) {
        if (polygon.size() >= 3) {
            // Calculate the area of the polygon
            double area = 0.0;
            for (auto vertex = polygon.vertices_begin(); vertex != polygon.vertices_end(); ++vertex) {
                auto next_vertex = std::next(vertex);
                if (next_vertex == polygon.vertices_end()) {
                    next_vertex = polygon.vertices_begin(); // Wrap around for the last edge
                }
                area += vertex->x() * next_vertex->y() - vertex->y() * next_vertex->x();
            }
            area = std::abs(area) / 2.0;
            if (area > image_area_threshold) {
                 // Calculate the center of the polygon
                cv::Point2f center(0, 0);
                for (auto vertex = polygon.vertices_begin(); vertex != polygon.vertices_end(); ++vertex) {
                    center.x += vertex->x();
                    center.y += vertex->y();
                }
                center.x /= polygon.size();
                center.y /= polygon.size();

                // Check if this polygon's center is too close to any existing center
                bool is_duplicate = false;
                for (const auto& existing_center : polygon_centers) {
                    if (cv::norm(center - existing_center) < center_distance_threshold) {
                        is_duplicate = true;
                        break;
                    }
                }

                if (!is_duplicate) {
                    // Add this polygon's center to the list of centers
                    polygon_centers.push_back(center);

                    // Convert CGAL polygon to OpenCV polygon
                    std::vector<cv::Point> cv_polygon;
                    for (auto vertex = polygon.vertices_begin(); vertex != polygon.vertices_end(); ++vertex) {
                        cv_polygon.emplace_back(vertex->x(), vertex->y());
                    }
                    polygons.push_back(cv_polygon);

                    // Draw the polygon on the result image
                    cv::polylines(result_image, cv_polygon, true, cv::Scalar(0, 255, 255), 3);
                }
            }
        }
    }

    // Visualize the result
    //cv::imshow("Binary Image", binary_image);
    cv::imshow("Detected Polygons", result_image);
    cv::waitKey(0);

    return polygons;
}

// Generate waypoints for a detected polygons. For multiple rooms and there is no wall inside
std::vector<Waypoint> generateWaypoints(const std::vector<std::vector<cv::Point>>& rectangles, float zCoordinate) {

    std::vector<Waypoint> all_waypoints;

    // Check if a waypoint is too close to any existing waypoint
    auto isTooCloseToExistingWaypoints = [&](const Waypoint& new_waypoint) -> bool {
        for (const auto& existing_waypoint : all_waypoints) {
            double distance = std::sqrt(std::pow(new_waypoint.x - existing_waypoint.x, 2) +
                                        std::pow(new_waypoint.y - existing_waypoint.y, 2));
            if (distance < points_min_distance) {
                return true; // Too close to an existing waypoint
            }
        }
        return false;
    };

    // Generate waypoints for all rectangles
    for (const auto& rectangle : rectangles) {
        std::vector<Waypoint> waypoints;

        // Transform rectangle vertices back to real-world coordinates
        auto real_world_rectangle = transformToRealWorld(rectangle, image_size);

        cv::Point2f center(0, 0);

        for (const auto& vertex : real_world_rectangle) {
            center += cv::Point2f(vertex.x, vertex.y); // Accumulate vertex positions
        }
        center *= (1.0 / real_world_rectangle.size()); // Compute the center

        bool repeat_first = true;
        Waypoint first_vertex;

        for (const auto& vertex : real_world_rectangle) {
            cv::Point2f vertex_f(vertex.x, vertex.y); // Convert to float point
            cv::Point2f scaled_vertex = center + (vertex_f - center) * 0.9;

            Waypoint new_waypoint = {scaled_vertex.x, scaled_vertex.y, zCoordinate};

            if (repeat_first) {
                first_vertex = new_waypoint;
                repeat_first = false;
            }

            if (!isTooCloseToExistingWaypoints(new_waypoint)) {
                    all_waypoints.push_back(new_waypoint);
            }
        }
        // Add waypoints to the global list
        all_waypoints.insert(all_waypoints.end(), waypoints.begin(), waypoints.end());
        all_waypoints.push_back(first_vertex);
    }
    return all_waypoints;
}


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

    // Check if a waypoint is too close to any existing waypoint
    auto isTooCloseToExistingWaypoints = [&](const Waypoint& new_waypoint) -> bool {
        for (const auto& existing_waypoint : all_waypoints) {
            double distance = std::sqrt(std::pow(new_waypoint.x - existing_waypoint.x, 2) +
                                        std::pow(new_waypoint.y - existing_waypoint.y, 2));
            if (distance < points_min_distance) {
                return true; // Too close to an existing waypoint
            }
        }
        return false;
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
                Waypoint new_waypoint = {scaled_vertex.x, scaled_vertex.y, zCoordinate};
                if (!isTooCloseToExistingWaypoints(new_waypoint)) {
                    all_waypoints.push_back(new_waypoint);
                }
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
        viewer->addSphere<pcl::PointXYZ>(pcl::PointXYZ(wp.x, wp.y, wp.z), 0.3, 0.0, 1.0, 1.0, point_id);
        waypoint_id++;
    }
}

void writeWaypointsToFile(const std::vector<Waypoint>& waypoints, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing");
    }
    for (const auto& waypoint : waypoints) {
        file << waypoint.x << " " << waypoint.y << " " << waypoint.z << "\n";
    }
    file.close();
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
        //Read the .pcd file and store into the cloud object (of type pcl::PointCloud<pcl::PointXYZ>).
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", file_path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu points from %s", cloud->size(), file_path.c_str());

        cv::Mat cloud_image = projectPointCloudToBinaryImage(cloud, image_size);
        //cv::imshow("Cloud image", cloud_image);

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
            //float d = coefficients->values[3];
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
        auto rectangles = detectShapes(wall_features, cloud_image);

        // Generate waypoints
        std::vector<Waypoint> all_waypoints;
        if (!walls_in_room) {
            all_waypoints = generateWaypoints(rectangles, z_coordinate);
        } else {
            all_waypoints = generateWaypointsFromRectangles(rectangles, z_coordinate); 
        }

        // Write waypoints to file
        writeWaypointsToFile(all_waypoints, "waypoints.txt");       

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