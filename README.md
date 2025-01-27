
# Automated Moblie Robot

This repository contains the code of implementation of a robot which can fully automated explore the environment while doing SLAM, it has also capability to process the point cloud from SLAM. 

Instructions for setting up and running the autonomous exploration framework with the following modules:

1. **Autonomy Basic**: [Autonomous Exploration Development Environment](https://github.com/HongbiaoZ/autonomous_exploration_development_environment)
2. **Far Planner**: [Far Planner](https://github.com/MichaelFYang/far_planner)
3. **LIO-SAM**: [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)
4. **TARE Planner**: [TARE Planner](https://github.com/caochao39/tare_planner)
5. **Superpoint Transformer**: [SPT](https://github.com/drprojects/superpoint_transformer/tree/master)

## Description
### 1. Exploration and SLAM
The robot starts by exploring the enviroment using AEDE; meanwhile LIO-SAM is running in the background to construct map of this environment.

### 2. Navigation
The AEDE provides basic autonomy (including coliision avoidance, traversability analysis and neighborhood navigation) and far_planner for distant goals in space.

### 3. Map processing
For the small and simple environment, RANSAC-menthod should be enough to detect all the walls in the building. For the large and complex environment, SPT is recomanded, to use it, build a conda environment on desktop at first, then use file transmitter to transfer the PLY from robot to desktop. It performs semantic segmentation for the point cloud on GPU, semantic information can be used for navigation, and help the robot to understand the environment. 

An example can be the following semantic map of an office, all objects with labels are abstacted to a polygon in the 2D map, the robot can access their positions by reading vertices of these polygons.

<p align="ledt">
  <img src="media/office semantic map.png" alt="Semantic map" height="500">
</p>

### 4. Waypoint generation
In this task, the robot is asked to inspect all the walls in the room, if you used RANSAC method in last step, the code will perform polygon dection for the map and generate waypoints by scale the polygons.

If you used SPT in last step, the robot will use lines to fit the features and generate waypoints for inspection. The false waypoint-lines will be filter out by using floor_mask. Then the waypoints will be reordered for better route planning. Following are some examples:

<div style="display: flex; gap: 20px;">
  <img src="media/test waypoints.png" alt="Image 1" style="height: 500px;">
  <img src="media/indoor waypoints.png" alt="Image 2" style="height: 500px;">
  <img src="media/test indoor waypoints.png" alt="Image 3" style="height: 500px;">
</div>

### 5. Control robot
Publish waypoints to corresponding ROS topic.

## Installation

### Dependencies

Install the following system dependencies:

```bash
sudo apt update
sudo apt install libusb-dev
sudo apt install ros-humble-perception-pcl        ros-humble-pcl-msgs        ros-humble-vision-opencv        ros-humble-xacro
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt install libgtsam-dev libgtsam-unstable-dev
sudo apt-get install libcgal-dev
```

### Step 1. Install the Required Modules

Clone the repositories for the modules:

```bash
git clone https://gitlab.lrz.de/00000000014B950F/autonomous-exploration-with-lio-sam.git
colcon build
```

### Step 2. Install TARE Planner

```bash
git clone https://github.com/caochao39/tare_planner.git
cd tare_planner
git checkout humble-jazzy
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Step 3. Install SPT
Make sure that your GPU is running with CUDA 11.8 or 12.1. For Ubuntu 22.04 with RTX 4070, follow this instruction https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba

Then clone the repo and install dependencies:
```bash
git clone https://github.com/drprojects/superpoint_transformer.git
cd spt
# Creates a conda env named 'spt' env and installs dependencies
./install.sh
```

If you have problems installing FRNN, try following:
```bash
git clone --recursive https://github.com/lxxue/FRNN.git
# For RTX 4070
export TORCH_CUDA_ARCH_LIST="8.9"
# install a prefix_sum routine first
cd FRNN/external/prefix_sum
pip install .

# install FRNN
cd ../../ # back to the {FRNN} directory
# For RTX 4070
export TORCH_CUDA_ARCH_LIST="8.9"
# this might take a while since I instantiate all combinations of D and K
pip install -e .
# might need to restart the bash to make importing this package work
```

## Usage

### 1. **Launch Gazebo Simulation and Autonomy Basic**:

   ```bash
   cd autonomous-exploration-with-lio-sam
   source install/setup.sh
   ros2 launch vehicle_simulator system_test.launch
   ```
   Or launch a different world:
   ```bash
   ros2 launch vehicle_simulator system_garage.launch
   ```
   Now you should see gazebo simulation and autonomy_basic visualized in RViz.

### 2. **Launch LIO-SAM**:

   ```bash
   cd autonomous-exploration-with-lio-sam
   source install/setup.sh
   ros2 launch lio_sam run.launch.py
   ```
   Now you should see LIO-SAM visualized in RViz.

### 3. **Launch tare_planner**:

   ```bash
   cd tare_planner
   source install/setup.sh
   ros2 launch tare_planner explore_garage.launch
   ```
   Now you should see autonomous exploration in action.

### 4. **Launch Map Processor (RANSAC method)**:

   ```bash
   source install/setup.bash
   ros2 run lidar_map_processor lidar_map_processor
   ```
   This module is developed for wall and ceiling detection and way point generating. Modify the pcd file path before using. 
   You should walls and ceiling are detected, after finihing matching, it will generates way points(marked in blue) around walls. Adjust scale_down and scale_up control the distance between way points and walls.

   RANSAC is used to fit the features of walls and ceiling in the point cloud, it finds the planes at first, which can filter out the curved surface like barrals or human in the enviroment. Then uses the normal vector to filter out other inrrevlent plane features like ladder in the room. Setting the area threshold for planes is optional, it may be helpful to filter out the inrrevlent some planes like toolbox in the room.

   Then CGAL is applied to detect the polygons in the binary image, which is the representaion of the environment.

   The RANSAC method may have difficulties dealing with large scale maps, since if the noise is too dense in the map, the RANSAC may evaluate the points (from walls and noise in the air) at same level as a plane which is parallel to floor, hence the walls can not be recognized. Need to crop the large cloud into multiple small clouds at first.It also have difficulties dealing with maps from different equipments(lidars). Need to fine-tune the distance parameters each time.

   **Attempt 1**: *(Finished)* use the functions of planes to draw lines that can be easily detected by CGAL while generator less noise.

   Result: NOT GOOD since any random point in space can be classifie to a wall as long as it's on the same plane, such outliers will make the boundary of a plane exetremly large.

   **Attempt 2**: *(Not needed)* use DL-based detection for polygon detection.

   Result: DL-based detetcion can be robust, but our goal here is generating coordinates of way points, which requires accurate bounding box of the wall, while neuron networks' bounding box is not accurate. But with some workaround it should work well.
   Mask R-CNN is an option, but running it requirs CPU(i7), GPU(RTX2080Ti), RAM(8G), the inference should be done on desktop or cloud if robot can send the point cloud file to them.

   **Attempt 3**: *(Current)* use image compressed from original cloud and a mask generated by filtered cloud to detection. Duplicate polygons and way points will be filter out.
   
   Result: If kernel size is proper, it works well.

   **Attempt 4**: *(Current)* Use neural network to handel point cloud and feature extraction directly. For this project, the superpoint_transformer(https://github.com/drprojects/superpoint_transformer/tree/master) and pretrained models(https://zenodo.org/records/8042712) are used. If only limited GPU memory is availiable(RTX 4070 in this case), check memory usage before starting; For large maps, use tiling at first to save memory.

### 5. **Superpoint transformer**:

   Before using, check if point cloud files are in correct format. For pcd files, use CloudCompare to convert and set RGB color. Check the coordinate in original files, to align z-axis with correct direction, in CloudCompare:

   Go to Edit -> Apply Transformation. In the transformation matrix dialog, set the values to reflect a flip in the Z-axis. This matrix effectively flips the Z-axis direction:
   ```bash
      1   0   0   0
      0   1   0   0
      0   0  -1   0
      0   0   0   1
   ```

   Also make sure that the ply files has following head:
   ```bash
   property float x
   property float y
   property float z
   property float red
   property float green
   property float blue
   ```

### 6. **Launch File Transmitter**:

   ```bash
   source install/setup.bash
   ros2 run file_transfer_pkg file_sender
   ros2 run file_transfer_pkg file_receiver
   ```
   Now you can transmit files from robot to desktop. 

### 7. **Launch Waypoint Publisher**:

   ```bash
   source install/setup.bash
   ros2 run waypoint_publisher waypoint_publisher
   ```
   Now you should see the robot moving around the features. 


## Others

1. The autonomy basic publish way points information to the /way_point topic.
   ```bash
   ros2 topic echo /way_point
   ```

2. To visulaize the .pcd file, you can use the online viewer:
https://imagetostl.com/view-pcd-online#convert