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

class LidarMapProcessor : public rclcpp::Node {
public:
    LidarMapProcessor() : Node("lidar_map_processor") {
        RCLCPP_INFO(this->get_logger(), "Starting Lidar Map Processor Node...");
        processPointCloud();
    }

private:
    void processPointCloud() {

      //double area_threshold = 10; // Area threshold in square meters
      double ignore_threshold = 0.001; // The value determines how many points of the whole cloud will be ignored
      double distance_threshold = 0.01;
      // Load the PCD file
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      std::string file_path = "/home/yuanyan/Downloads/LOAM/cloudGlobal.pcd";
      //Read the .pcd file and store into the cloud object (of type pcl::PointCloud<pcl::PointXYZ>).
      if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
          RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", file_path.c_str());
          return;
      }
      RCLCPP_INFO(this->get_logger(), "Loaded %lu points from %s", cloud->size(), file_path.c_str());

      // Prepare visualizer
      pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Extracted Planes"));
      viewer->setBackgroundColor(0, 0, 0);
      viewer->addPointCloud<pcl::PointXYZ>(cloud, "original_cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");

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

          // Estimate the area of the plane
          /*
          pcl::PointXYZ min_pt, max_pt;
          pcl::getMinMax3D(*plane_cloud, min_pt, max_pt);
          double plane_area = (max_pt.x - min_pt.x) * (max_pt.y - min_pt.y);

          if (plane_area < area_threshold) {
            RCLCPP_INFO(this->get_logger(), "Plane %d discarded: area %.2f m² below threshold %.2f m²",
                        i, plane_area, area_threshold);
            extract.setNegative(true);
            extract.filter(*cloud_filtered);
            cloud.swap(cloud_filtered);
            continue;
          }

          RCLCPP_INFO(this->get_logger(), "Plane %d: area %.2f m²", i, plane_area);
          */

          // Determine if the plane is a wall or ceiling
          Eigen::Vector4f centroid;
          pcl::compute3DCentroid(*plane_cloud, centroid);

          float a = coefficients->values[0];
          float b = coefficients->values[1];
          float c = coefficients->values[2];
          float d = coefficients->values[3];
          Eigen::Vector3f normal(a, b, c);

          /*
          // Check orientation of the plane
          if (std::abs(normal.dot(Eigen::Vector3f(0, 0, 1))) < 0.1) {
              RCLCPP_INFO(this->get_logger(), "Detected vertical wall at centroid (%.2f, %.2f, %.2f)",
                          centroid[0], centroid[1], centroid[2]);
          } else if (normal.dot(Eigen::Vector3f(0, 0, -1)) > 0.9) {
              RCLCPP_INFO(this->get_logger(), "Detected horizontal ceiling at centroid (%.2f, %.2f, %.2f)",
                          centroid[0], centroid[1], centroid[2]);
          } else {
              RCLCPP_INFO(this->get_logger(), "Other plane detected.");
          }
          */

          std::string plane_name = "Plane_" + std::to_string(i);
          if (std::abs(normal.dot(Eigen::Vector3f(0, 0, 1))) < 0.1) {
              RCLCPP_INFO(this->get_logger(), "Detected vertical wall at centroid (%.2f, %.2f, %.2f)",
                          centroid[0], centroid[1], centroid[2]);
              viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
              viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                        1.0, 0.0, 0.0, plane_name); //Showed in red
          } else if (normal.dot(Eigen::Vector3f(0, 0, -1)) > 0.9) {
              RCLCPP_INFO(this->get_logger(), "Detected horizontal ceiling at centroid (%.2f, %.2f, %.2f)",
                          centroid[0], centroid[1], centroid[2]);
              viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
              viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                        0.0, 1.0, 0.0, plane_name); //Showed in blue
          } else {
              RCLCPP_INFO(this->get_logger(), "Other plane detected.");
          }

          // Remove the extracted plane from the cloud
          extract.setNegative(true);
          extract.filter(*cloud_filtered);
          cloud.swap(cloud_filtered);
          i++;
        }

        // Save the processed file as a .ply
        std::string output_file = "/home/yuanyan/autonomous-exploration-with-lio-sam/src/vehicle_simulator/mesh/test/preview/pointcloud.ply";
        if (pcl::io::savePLYFile(output_file, *cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't write file: %s", output_file.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Processed cloud saved as: %s", output_file.c_str());
        }

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