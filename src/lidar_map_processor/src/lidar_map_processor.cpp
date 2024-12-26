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
std::string file_path = "test.pcd";

// Mode selection
bool walls_in_room = false; // false means there is no individual walls in the room

// RANSAC parameters
const double ignore_threshold = 0.015; // The value determines how many points of the whole cloud will be ignored.
const double distance_threshold = 0.05; // Too large is also not good, 0.05

// CV parameters
const int image_size = 200; // Adjust based on the map size, calculate the length each pixel represents in real world. It will also affect the kernel size.
const float z_coordinate = 0.1;
const float image_area_threshold = 500.0; //should be dynamic with image size, can't use area in point cloud, since a random point in space can still be classifie to a wall as long as it's on the same plane.

const float clearance = 1.5; // The distance between way points and walls in the room.
const float n_clearance = -0.6; // The distance between way points and walls in the room.
const double center_distance_threshold = 5; // Filter out duplicate polygons, 2

float min_x, max_x, min_y, max_y;
float width, height, aspect_ratio;

// Way point generator
const double points_min_distance = 0.5; // meters, filter out duplicate way points

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
    cv::imshow("sparse_image", sparse_image);
    cv::imshow("dense_image", dense_image);

    cv::Mat denoise_dense;
    cv::GaussianBlur(dense_image, denoise_dense, cv::Size(3, 3), 0); // better than medianBlur

    // Dilate the sparse image to create a region of interest (ROI)
    cv::Mat roi_mask;
    cv::Mat kernel_mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)); 
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
    cv::Mat kernel_mask_2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 4)); 
    cv::Mat kernel_mask_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 1)); 
    cv::dilate(refined_image, output_image, kernel_mask_2);
    cv::dilate(output_image, output_image, kernel_mask_3);
        
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // Kernel size, 5 for test.world
    //cv::morphologyEx(refined_image, output_image, cv::MORPH_CLOSE, kernel);

    // Save the output image
    //cv::imwrite("output_image.png", output_image);
    //cv::imshow("output_img", output_image);
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
        double epsilon = 0.02 * cv::arcLength(contour, true); // Adjust epsilon for simplification, 0.02 for test.world
        cv::approxPolyDP(contour, simplified_contour, epsilon, true);

        // Convert OpenCV contour to CGAL points
        std::vector<Point_2> contour_points;
        for (const auto& pt : simplified_contour) {
            contour_points.emplace_back(pt.x, pt.y); // Use image coordinates directly
        }

        // Visualize the raw contour
        cv::drawContours(result_image, std::vector<std::vector<cv::Point>>{simplified_contour}, -1, cv::Scalar(255, 255, 255), 1);

        Polygon_2 polygon(contour_points.begin(), contour_points.end());
        // Check if the polygon is valid
        if (polygon.size() >= 4) {
            // Calculate the area of the polygon
            double area = abs(polygon.area());
            
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
                    cv::polylines(result_image, cv_polygon, true, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
        
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
        Polygon_2 convex_polygon(hull.begin(), hull.end());
        // Check if the polygon is valid
        if (convex_polygon.size() >= 4) {
            // Calculate the area of the polygon
            double area = 0.0;
            for (auto vertex = convex_polygon.vertices_begin(); vertex != convex_polygon.vertices_end(); ++vertex) {
                auto next_vertex = std::next(vertex);
                if (next_vertex == convex_polygon.vertices_end()) {
                    next_vertex = convex_polygon.vertices_begin(); // Wrap around for the last edge
                }
                area += vertex->x() * next_vertex->y() - vertex->y() * next_vertex->x();
            }
            area = std::abs(area) / 2.0;
            if (area > image_area_threshold) {
                 // Calculate the center of the polygon
                cv::Point2f convex_center(0, 0);
                for (auto vertex = convex_polygon.vertices_begin(); vertex != convex_polygon.vertices_end(); ++vertex) {
                    convex_center.x += vertex->x();
                    convex_center.y += vertex->y();
                }
                convex_center.x /= convex_polygon.size();
                convex_center.y /= convex_polygon.size();

                // Check if this polygon's center is too close to any existing center
                bool is_duplicate = false;
                for (const auto& existing_center : polygon_centers) {
                    if (cv::norm(convex_center - existing_center) < center_distance_threshold) {
                        is_duplicate = true;
                        break;
                    }
                }

                if (!is_duplicate) {
                    // Add this polygon's center to the list of centers
                    polygon_centers.push_back(convex_center);

                    // Convert CGAL polygon to OpenCV polygon
                    std::vector<cv::Point> cv_convex_polygon;
                    for (auto vertex = convex_polygon.vertices_begin(); vertex != convex_polygon.vertices_end(); ++vertex) {
                        cv_convex_polygon.emplace_back(vertex->x(), vertex->y());
                    }
                    polygons.push_back(cv_convex_polygon);

                    // Draw the polygon on the result image
                    cv::polylines(result_image, cv_convex_polygon, true, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
        */
    }

    // Visualize the result
    cv::imshow("Binary Image", binary_image);
    cv::imshow("Detected Polygons", result_image);
    cv::waitKey(0);

    return polygons;
}

double pointToSegmentDistance(const cv::Point& point, const cv::Point& line_start, const cv::Point& line_end) {
    double x0 = point.x, y0 = point.y;
    double x1 = line_start.x, y1 = line_start.y;
    double x2 = line_end.x, y2 = line_end.y;

    double A = x0 - x1, B = y0 - y1, C = x2 - x1, D = y2 - y1;

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    double param = (len_sq != 0) ? dot / len_sq : -1;

    double xx, yy;

    if (param < 0) {
        xx = x1;
        yy = y1;
    } else if (param > 1) {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }

    double dx = x0 - xx;
    double dy = y0 - yy;
    return std::sqrt(dx * dx + dy * dy);
}

// Function to determine if any vertex of a polygon lies on the edge of another polygon
bool isVertexOnEdge(const std::vector<std::vector<cv::Point>>& polygons, double threshold) {
    for (size_t i = 0; i < polygons.size(); ++i) {
        for (size_t j = 0; j < polygons.size(); ++j) {
            if (i == j) continue; // Skip checking the polygon against itself

            const auto& poly1 = polygons[i];
            const auto& poly2 = polygons[j];

            // Check each vertex of poly1 against all edges of poly2
            for (const auto& vertex : poly1) {
                for (size_t k = 0; k < poly2.size(); ++k) {
                    cv::Point edge_start = poly2[k];
                    cv::Point edge_end = poly2[(k + 1) % poly2.size()]; // Wrap around to the first point

                    // Calculate the distance from the vertex to the edge
                    double distance = pointToSegmentDistance(vertex, edge_start, edge_end);

                    if (distance < threshold) {
                        return false; // A vertex of poly1 is close to an edge of poly2
                    }
                }
            }
        }
    }
    return true;
}

// Generate waypoints for a detected polygons. For multiple rooms and there is no wall inside
std::vector<Waypoint> generateWaypoints(const std::vector<std::vector<cv::Point>>& rectangles, float zCoordinate) {

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
        bool is_largest_polygon = (&rectangle == largest_rectangle);

        if (!is_largest_polygon) {
            std::vector<Waypoint> waypoints;

            // Transform rectangle vertices back to real-world coordinates
            auto real_world_rectangle = transformToRealWorld(rectangle, image_size);

            size_t n = real_world_rectangle.size(); // Number of vertices
            for (size_t i = 0; i < n; ++i) {
                // Get previous, current, and next vertices
                cv::Point2f prev_vertex = real_world_rectangle[(i + n - 1) % n];
                cv::Point2f curr_vertex = real_world_rectangle[i];
                cv::Point2f next_vertex = real_world_rectangle[(i + 1) % n];

                // Compute vectors from the current vertex to the adjacent vertices
                cv::Point2f v1 = prev_vertex - curr_vertex; // Vector to previous vertex
                cv::Point2f v2 = next_vertex - curr_vertex; // Vector to next vertex

                // Normalize edge vectors
                v1 /= std::sqrt(v1.x * v1.x + v1.y * v1.y);
                v2 /= std::sqrt(v2.x * v2.x + v2.y * v2.y);


                // Compute the angular bisector by summing normalized vectors
                cv::Point2f bisector = v1 + v2;
                float bisector_length = std::sqrt(bisector.x * bisector.x + bisector.y * bisector.y);
                if (bisector_length != 0) {
                    bisector /= bisector_length; // Normalize bisector
                }

                // Compute the waypoint by moving along the bisector from the current vertex
                cv::Point2f waypoint_position = curr_vertex + clearance * bisector;

                // Create the waypoint
                Waypoint new_waypoint = {waypoint_position.x, waypoint_position.y, zCoordinate};

                // Check if the waypoint is valid and add it
                if (!isTooCloseToExistingWaypoints(new_waypoint)) {
                    waypoints.push_back(new_waypoint);
                }
            }

            // Add waypoints to the global list
            all_waypoints.insert(all_waypoints.end(), waypoints.begin(), waypoints.end());
        }
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

    for (const auto& rectangle : rectangles) {
        bool is_largest_polygon = (&rectangle == largest_rectangle);
        float distance = is_largest_polygon ? clearance : n_clearance;
        
        std::vector<Waypoint> waypoints;

        // Transform rectangle vertices back to real-world coordinates
        auto real_world_rectangle = transformToRealWorld(rectangle, image_size);

        size_t n = real_world_rectangle.size(); // Number of vertices
        for (size_t i = 0; i < n; ++i) {
            // Get previous, current, and next vertices
            cv::Point2f prev_vertex = real_world_rectangle[(i + n - 1) % n];
            cv::Point2f curr_vertex = real_world_rectangle[i];
            cv::Point2f next_vertex = real_world_rectangle[(i + 1) % n];

            // Compute vectors from the current vertex to the adjacent vertices
            cv::Point2f v1 = prev_vertex - curr_vertex; // Vector to previous vertex
            cv::Point2f v2 = next_vertex - curr_vertex; // Vector to next vertex

            // Normalize these vectors
            v1 /= std::sqrt(v1.x * v1.x + v1.y * v1.y);
            v2 /= std::sqrt(v2.x * v2.x + v2.y * v2.y);

            // Compute the angular bisector by summing normalized vectors
            cv::Point2f bisector = v1 + v2;
            float bisector_length = std::sqrt(bisector.x * bisector.x + bisector.y * bisector.y);
            if (bisector_length != 0) {
                bisector /= bisector_length; // Normalize bisector
            }

            // Compute the waypoint by moving along the bisector from the current vertex
            cv::Point2f waypoint_position = curr_vertex + distance * bisector;

            // Create the waypoint
            Waypoint new_waypoint = {waypoint_position.x, waypoint_position.y, zCoordinate};

            // Check if the waypoint is valid and add it
            if (!isTooCloseToExistingWaypoints(new_waypoint)) {
                waypoints.push_back(new_waypoint);
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

            } else if (normal.dot(Eigen::Vector3f(0, 0, -1)) > 0.95) {
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

        walls_in_room = isVertexOnEdge(rectangles, 5.0);

        // Generate waypoints
        std::vector<Waypoint> all_waypoints;
        if (!walls_in_room) {
            RCLCPP_INFO(this->get_logger(), "walls_in_room: false");
            all_waypoints = generateWaypoints(rectangles, z_coordinate);
        } else {
            RCLCPP_INFO(this->get_logger(), "walls_in_room: true");
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