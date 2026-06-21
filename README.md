# UKF on Manifold

A C++ implementation of an **Unscented Kalman Filter (UKF) on Lie manifolds** for inertial navigation and sensor fusion.

The project provides implementations and examples for:

* GNSS–Inertial Odometry (IMU + GPS)
* LiDAR–Inertial Odometry (UKF + MAD-ICP)
* IMU bias estimation
* Gravity estimation
* Extrinsic calibration estimation
* Synthetic trajectory generation
* Filter evaluation using ATE and RPE metrics

The theoretical background is described in **`ukf_manifolds_notes.pdf`**.

---

## Features

* UKF operating directly on Lie manifolds
* SO(3) state representation
* IMU propagation model
* GPS measurement update
* LiDAR-Inertial Odometry pipeline
* MAD-ICP integration
* Support for IMU bias estimation
* Gravity estimation
* Synthetic datasets for testing
* Performance evaluation tools

---

## Project Structure

```
.
├── include/
├── src/
├── examples/
├── datasets/
├── build/
├── ukf_manifolds_notes.pdf
└── README.md
```

---

## Dependencies

The project requires:

* C++17
* CMake
* Eigen3

Install Eigen:

```bash
sudo apt install libeigen3-dev
```

---

## Build

Clone the repository and compile the project:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

---

# GNSS–Inertial Odometry

Several examples are provided to test the UKF using synthetic IMU and GPS data.

Move to the examples directory:

```bash
cd build/examples
```

Run the main example:

```bash
./example_complete <output_prefix> [trajectory_type] [trajectory_amplitude] [bias_multiplier]
```

Example:

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

## Initialization Analysis

To compare different initialization strategies:

```bash
./example_initialization ~/Desktop/traj
```

This executable requires an EuRoC dataset.

Download a compatible dataset and update the dataset path inside the source file before compiling.

---

## Filter Evaluation

This example compares different UKF configurations, including filters with and without IMU bias estimation.

```bash
./example_evaluation_metrics <output_prefix> <trajectory_type> <variation_type>
```

Example:

```bash
./example_evaluation_metrics ~/Desktop/traj 1 amplitude
```

---

# LiDAR–Inertial Odometry

The project also includes a LiDAR-Inertial Odometry implementation combining UKF and MAD-ICP.

Run:

```bash
./example_UKF_NEWCO <output_prefix> <trajectory_type>
```

Example:

```bash
./example_UKF_NEWCO ~/Desktop/UKF_ICP 1
```

The trajectory type selects one of the supported ROS2 datasets.

Download a compatible dataset and update the dataset path inside the source file before compiling.

---

## Complete LiDAR-Inertial Pipeline

The executable

```bash
./LIO-UKF-MADICP <output_prefix> <trajectory_type>
```

contains the complete pipeline, including:

* IMU propagation
* Point cloud undistortion
* MAD-ICP registration
* UKF correction

The file `example_UKF_NEWCO.cpp` only demonstrates the fusion between UKF and MAD-ICP without point cloud undistortion.

---

# Output Files

Depending on the executable, the generated files may include:

* Estimated trajectory
* Ground-truth trajectory
* Filter states
* ATE metrics
* RPE metrics
* Evaluation statistics

Trajectory files may contain only the global position:

```
tx ty tz
```

Or a TUM format position:

```
dt tx ty tz qx qy qz qw
```

---

# Visualization

The generated trajectories can be visualized using any plotting software.

A simple option is **gnuplot**.

Launch:

```bash
gnuplot
```

Example:

```gnuplot
splot "traj_est.txt" using 1:2:3 with lines, \
      "traj_gt.txt" using 1:2:3 with lines
```

To compare multiple trajectories:

```gnuplot
splot \
"traj_gt.txt" using 1:2:3 with lines, \
"traj_est_ext_bias.txt" using 1:2:3 with lines, \
"traj_est_bias.txt" using 1:2:3 with lines
```

To compare evaluation metrics:

```gnuplot
plot \
"ATE-RPE_results_NavExt.txt" using 2 with lines, \
"ATE-RPE_results_Nav.txt" using 2 with lines
```

Adjust the filenames according to the generated output.

---

# Future Work

Planned improvements include:

* Command-line dataset selection
* Configuration through YAML files
* ROS2 package
* RViz visualization
* Automatic dataset discovery
* Additional sensor models

---

# References

If you use this project, please refer to:

* `ukf_manifolds_notes.pdf`
* Relevant literature on UKF on Lie Groups and Manifolds

