# UKF-based Multi-Sensor Odometry

A C++ framework for **state estimation on Lie manifolds** using an **Unscented Kalman Filter (UKF)**, with applications to **LiDAR–Inertial Odometry (LIO)** and **GNSS–Inertial Odometry (GIO)**.

The project focuses on the implementation of a UKF operating directly on manifold-valued states and its integration with multiple sensing modalities. The main application is a complete **LiDAR–Inertial Odometry pipeline**, combining IMU propagation, point cloud undistortion, MAD-ICP registration, and UKF state estimation. The repository also includes examples for GNSS-aided inertial navigation using synthetic data.

The algorithm and its practical implementation details are described in **"Master_Thesis.pdf"**.

The theoretical background about UKF is described in **`ukf_manifolds_notes.pdf`**.

The theoretical background about MAD-ICP is described in **`MAD-ICP: It Is All About Matching Data--Robust and Informed LiDAR Odometry`**.

---

# Repository Organization

The repository is organized into two independent components.

## 1. Core C++ Implementation

This is the main project and contains the complete implementation of the filtering framework. It can be executed directly without ROS and includes:

* GNSS–Inertial Odometry (IMU + GPS)
* LiDAR–Inertial Odometry (UKF + MAD-ICP)
* IMU bias estimation
* Gravity estimation
* Extrinsic calibration estimation
* Synthetic trajectory generation
* Evaluation using ATE and RPE metrics

This version reads datasets directly from disk, executes the complete estimation pipeline, and generates trajectories and evaluation files.

---

## 2. ROS2 Visualization Package

A second folder contains a ROS2 package developed to visualize the LiDAR-Inertial pipeline in **RViz2**.

Its main purpose is visualization. It publishes the data produced by the odometry pipeline, including point clouds, poses, trajectories, and other debugging information, allowing the internal behavior of the algorithm to be inspected in real time.

The ROS2 package is independent from the core implementation and is intended as a visualization and debugging tool.

---

# Features

* Unscented Kalman Filter on Lie manifolds
* SO(3)-based state representation
* IMU propagation model
* GPS measurement update
* LiDAR–Inertial Odometry
* GNSS–Inertial Odometry
* MAD-ICP registration
* Point cloud undistortion
* IMU bias estimation
* Gravity estimation
* Extrinsic calibration estimation
* Synthetic dataset generation
* Performance evaluation using ATE and RPE
* ROS2 visualization support through RViz2

---

# Project Structure

```
.
├── core/                  # Standalone C++ implementation
│   ├── include/
│   ├── src/
│   ├── examples/
│   ├── datasets/
│   └── ...
│
├── ros2/                  # ROS2 visualization package
│   ├── src/
│   ├── launch/
│   ├── rviz/
│   └── ...
│
├── ukf_manifolds_notes.pdf
└── README.md
```

---

# Dependencies

The standalone implementation requires:

* C++17
* CMake
* Eigen3

Install Eigen:

```bash
sudo apt install libeigen3-dev
```

The ROS2 package additionally requires:

* ROS2 Jazzy
* PCL
* RViz2
* sensor_msgs
* pcl_conversions
* rosbag2_cpp

---

# Build

## Core implementation

```bash
mkdir build
cd build
cmake ..
make -j
```

---

## ROS2 package

Build the package inside a ROS2 workspace:

```bash
colcon build --symlink-install
source install/setup.bash
```
Open the visualizer in another terminal:

```bash
rviz2
```

Then launch the UKF node:

```bash
ros2 run ukf_lio_rviz lio_rviz
```

---

# GNSS–Inertial Odometry

Several examples are provided to test the UKF using synthetic IMU and GPS data.

Move to

```bash
cd build/examples
```

Run

```bash
./example_complete <output_prefix> [trajectory_type] [trajectory_amplitude] [bias_multiplier]
```

Example

```bash
./example_complete ~/Desktop/traj
```

or

```bash
./example_complete ~/Desktop/traj 1 10 1-1-1-1
```

### Parameters

| Parameter            | Description                          |
| -------------------- | ------------------------------------ |
| output_prefix        | Prefix of the generated output files |
| trajectory_type      | Synthetic trajectory type            |
| trajectory_amplitude | Amplitude of the trajectory          |
| bias_multiplier      | Artificial IMU bias multiplier       |

---

# Initialization Analysis

To compare different initialization strategies:

```bash
./example_initialization ~/Desktop/traj
```

This executable requires a EuRoC dataset.

Update the dataset path before compilation.

---

# Filter Evaluation

Compare different UKF configurations:

```bash
./example_evaluation_metrics <output_prefix> <trajectory_type> <variation_type>
```

Example

```bash
./example_evaluation_metrics ~/Desktop/traj 1 amplitude
```

---

# LiDAR–Inertial Odometry

The project includes two executables.

## UKF + MAD-ICP

```bash
./example_UKF_NEWCO <output_prefix> <trajectory_type>
```

This example demonstrates the fusion between the UKF and MAD-ICP.

---

## Complete LiDAR-Inertial Pipeline

```bash
./LIO-UKF-MADICP <output_prefix> <trajectory_type>
```

This executable performs the complete estimation pipeline:

* IMU propagation
* Point cloud undistortion
* MAD-ICP registration
* UKF correction

The trajectory type selects one of the supported datasets.

Update the dataset path before compiling.

---

# Output Files

Depending on the executable, the generated outputs include:

* Estimated trajectory
* Ground-truth trajectory
* Estimated states
* ATE metrics
* RPE metrics
* Evaluation statistics

Trajectories are exported either as

```
tx ty tz
```

or in TUM format

```
timestamp tx ty tz qx qy qz qw
```

---

# Visualization

The standalone implementation exports trajectories that can be visualized with any plotting software.

For example, using **gnuplot**:

```gnuplot
splot "traj_est.txt" using 1:2:3 with lines,\
      "traj_gt.txt" using 1:2:3 with lines
```

The ROS2 package provides real-time visualization in RViz2, including:

* undistorted point clouds
* estimated trajectory
* poses
* debugging information

---

# Future Work

* YAML-based configuration
* Automatic dataset discovery
* Online parameter tuning
* Additional LiDAR registration methods
* Additional sensor models
* Full ROS2 integration of the estimation pipeline

---

# References

If you use this repository, please cite or refer to:

* `ukf_manifolds_notes.pdf`
* `MAD-ICP: It Is All About Matching Data--Robust and Informed LiDAR Odometry`
