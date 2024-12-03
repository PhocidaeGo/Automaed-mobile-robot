
# Autonomous Exploration with LIO-SAM

This repository contains the code and instructions for setting up and running the autonomous exploration framework with the following modules:

1. **Autonomy Basic**: [Autonomous Exploration Development Environment](https://github.com/HongbiaoZ/autonomous_exploration_development_environment)
2. **Far Planner**: [Far Planner](https://github.com/MichaelFYang/far_planner)
3. **LIO-SAM**: [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)
4. **TARE Planner**: [TARE Planner](https://github.com/caochao39/tare_planner)

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

### Step 1: Install the Required Modules

Clone the repositories for the modules:

```bash
git clone https://gitlab.lrz.de/00000000014B950F/autonomous-exploration-with-lio-sam.git
colcon build
```

### Step 2: Install TARE Planner

```bash
git clone https://github.com/caochao39/tare_planner.git
cd tare_planner
git checkout humble-jazzy
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Usage

1. **Launch Gazebo Simulation and Autonomy Basic**:

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

2. **Launch LIO-SAM**:

   ```bash
   cd autonomous-exploration-with-lio-sam
   source install/setup.sh
   ros2 launch lio_sam run.launch.py
   ```
   Now you should see LIO-SAM visualized in RViz.

3. **Launch tare_planner**:

   ```bash
   cd tare_planner
   source install/setup.sh
   ros2 launch tare_planner explore_garage.launch
   ```
   Now you should see autonomous exploration in action.

4. **Launch Map Processor**:

   ```bash
   source install/setup.bash
   ros2 run lidar_map_processor lidar_map_processor
   ```
   Check if all the walls and ceiling are correctly segmented


## Others

1. The autonomy basic publish way points information to the /way_point topic.
   ```bash
   ros2 topic echo /way_point
   ```

2. To visulaize the .pcd file, you can use the online viewer:
https://imagetostl.com/view-pcd-online#convert

3. For the QR-code detection and coverage check in the construction site, RANSAC and CGAL are applied.

RANSAC is used to fit the features of walls and ceiling in the point cloud, it finds the planes at first, which can filter out the curved surface like barrals or human in the enviroment. Then uses the normal vector to filter out other inrrevlent plane features like ladder in the room. Setting the area threshold for planes is optional, it may be helpful to filter out the inrrevlent some planes like toolbox in the room.

Then CGAL is applied to detect the polygons in the binary image, which is the representaion of the environment. Furthermore, we can use the functions of planes to draw lines that can be easily detected by CGAL while generator less noise.do
