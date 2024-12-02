
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

The autonomy basic publish way points information to the /way_point topic.
Example:
ros2 topic echo /way_point
header:
  stamp:
    sec: 1733154718
    nanosec: 612243577
  frame_id: map
point:
  x: 3.597280263900757
  y: 2.663806438446045
  z: 0.7960537672042847
---

To visulaize the .pcd file, you can use the online viewer:
https://imagetostl.com/view-pcd-online#convert
