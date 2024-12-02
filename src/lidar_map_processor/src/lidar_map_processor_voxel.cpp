#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>

class LidarMapProcessor : public rclcpp::Node {
public:
    LidarMapProcessor() : Node("lidar_map_processor"), area_threshold_(1.0) {
        RCLCPP_INFO(this->get_logger(), "Starting Lidar Map Processor Node...");
        this->declare_parameter("area_threshold", 1.0); // Default threshold: 1.0 m²
        this->get_parameter("area_threshold", area_threshold_);
        RCLCPP_INFO(this->get_logger(), "Using area threshold: %.2f m²", area_threshold_);
        processPointCloud();
    }

private:
    double area_threshold_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f); // Adjust voxel size as needed
        voxel_filter.filter(*filtered_cloud);
        return filtered_cloud;
    }

    std::vector<pcl::PointIndices> regionGrowingSegmentation(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        pcl::PointCloud<pcl::Normal>::Ptr normals) {
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize(100);
        reg.setMaxClusterSize(10000);
        reg.setSearchMethod(pcl::search::Search<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>()));
        reg.setNumberOfNeighbours(30);
        reg.setInputCloud(cloud);
        reg.setInputNormals(normals);
        reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI); // 3 degrees
        reg.setCurvatureThreshold(1.0);

        std::vector<pcl::PointIndices> clusters;
        reg.extract(clusters);
        return clusters;
    }

    void processPointCloud() {
        // Load the PCD file
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        std::string file_path = "/home/yuanyan/Downloads/LOAM/cloudGlobal.pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", file_path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu points from %s", cloud->size(), file_path.c_str());

        // Preprocess the cloud to densify sparse data
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = preprocessPointCloud(cloud);

        // Estimate normals for region-growing segmentation
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(filtered_cloud);
        ne.setSearchMethod(pcl::search::Search<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>()));
        ne.setRadiusSearch(0.1);
        ne.compute(*normals);

        // Perform region-growing segmentation
        std::vector<pcl::PointIndices> clusters = regionGrowingSegmentation(filtered_cloud, normals);

        // Visualize the extracted planes
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Extracted Planes"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(cloud, "original_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");

        for (size_t i = 0; i < clusters.size(); ++i) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*filtered_cloud, clusters[i].indices, *plane_cloud);

            // Estimate the area of the plane
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*plane_cloud, min_pt, max_pt);
            double plane_area = (max_pt.x - min_pt.x) * (max_pt.y - min_pt.y);

            if (plane_area < area_threshold_) {
                RCLCPP_INFO(this->get_logger(), "Plane %zu discarded: area %.2f m² below threshold %.2f m²",
                            i, plane_area, area_threshold_);
                continue;
            }

            RCLCPP_INFO(this->get_logger(), "Plane %zu: area %.2f m²", i, plane_area);

            // Determine if the plane is a wall or ceiling
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*plane_cloud, centroid);

            Eigen::Vector3f normal(
                normals->points[clusters[i].indices[0]].normal_x,
                normals->points[clusters[i].indices[0]].normal_y,
                normals->points[clusters[i].indices[0]].normal_z);

            std::string plane_name = "Plane_" + std::to_string(i);
            if (std::abs(normal.dot(Eigen::Vector3f(0, 0, 1))) < 0.1) {
                RCLCPP_INFO(this->get_logger(), "Detected vertical wall at centroid (%.2f, %.2f, %.2f)",
                            centroid[0], centroid[1], centroid[2]);
                viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                         1.0, 0.0, 0.0, plane_name);
            } else if (normal.dot(Eigen::Vector3f(0, 0, -1)) > 0.9) {
                RCLCPP_INFO(this->get_logger(), "Detected horizontal ceiling at centroid (%.2f, %.2f, %.2f)",
                            centroid[0], centroid[1], centroid[2]);
                viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                         0.0, 1.0, 0.0, plane_name);
            } else {
                RCLCPP_INFO(this->get_logger(), "Other plane detected.");
                viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                         0.0, 0.0, 1.0, plane_name);
            }
        }

        // Save the processed file as a .ply
        std::string output_file = "/home/yuanyan/Downloads/LOAM/cloudGlobal.ply";
        pcl::io::savePLYFile(output_file, *filtered_cloud);
        RCLCPP_INFO(this->get_logger(), "Processed cloud saved as: %s", output_file.c_str());

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
