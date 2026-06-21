#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>
#include <ukf_manifold/lie_algebra.h>

#include <odometry/mad_icp.h>
#include <odometry/pipeline.h>
#include <odometry/vel_estimator.h>

#include <tools/constants.h>

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <cmath>

Eigen::Vector3d rotMatToEuler(const Eigen::Matrix3d& R){
  Eigen::Vector3d euler;
  double sy = std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0));
  bool singular = sy < 1e-6;
  if (!singular) {
    euler[0] = std::atan2(R(2,1), R(2,2));  // roll
    euler[1] = std::atan2(-R(2,0), sy);     // pitch
    euler[2] = std::atan2(R(1,0), R(0,0));  // yaw
  }
  else {
    // Gimbal lock
    euler[0] = std::atan2(-R(1,2), R(1,1)); // roll
    euler[1] = std::atan2(-R(2,0), sy);     // pitch
    euler[2] = 0;                           // yaw
  }
  return euler; // [roll, pitch, yaw]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// -------------------------------------------- Utilities for usage: -----------------------------------------
//
// ./example_UKF_NEWCO /home/giordano/Desktop/UKF_ICP 1
//
// ./this-executable path-to-output trajectory-type
//
// ------------------------------------------- Utilities for plotting: ----------------------------------------
//
//  splot "gt_states.txt" using 1:2:3 w l, "est_1.txt" using 1:2:3 w l, "est_2.txt" using 1:2:3 w l, "est_3.txt" using 1:2:3 w l
//
//  splot "est_lio.txt" using 1:2:3 w l
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ukf_manifold;

const std::string data_folder(UKF_DATA_FOLDER);

using ImuStates        = std::vector<NavState>;
using ImuStatesExt     = std::vector<NavStateExtended>;
using ImuLidarStates   = std::vector<NavStateLidar>;
using ImuInputs        = std::vector<Vector6d>;
using GpsMeasurements  = std::map<int, Eigen::Vector3d>;
using IntegrationSteps = std::vector<double>; 

using Vector4d    = Eigen::Matrix<double, 4, 1>;
using Matrix12d   = Eigen::Matrix<double, 12, 12>;

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;
using ContainerType = std::vector<Eigen::Vector3d>;
struct LidarMeasurement {
  double timestamp;
  ContainerType cloud;
};
using LidarInputs = std::vector<LidarMeasurement>;

void read_ros2_bag_imu(
                      IntegrationSteps&  imu_dts_, 
                      ImuInputs&         imu_inputs_, 
                      LidarInputs&       lidar_inputs_, 
                      const std::string& bag_path){

  std::cout << "... reading the database ...\n";
  rosbag2_cpp::Reader reader;
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path;
  storage_options.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";
  reader.open(storage_options, converter_options);
  rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> lidar_serialization;
  int imu = 0, lid = 0;
  std::cout << std::fixed << std::setprecision(9);

  // scrittura su txt dei timestamps delle misure IMU + LIDAR
  std::ofstream f_imu("/home/giordano/Desktop/UKF_ICP/imu_measures_times.txt");
  if (!f_imu.is_open()) {
    std::cerr << "Error opening file: " << "/home/giordano/Desktop/UKF_ICP/imu_measures_times.txt" << std::endl;
    return;
  }
  f_imu << "timestamp   presence(yes/no) " << "\n";
  f_imu << std::fixed << std::setprecision(9);
  std::ofstream f_lid("/home/giordano/Desktop/UKF_ICP/lid_measures_times.txt");
  if (!f_lid.is_open()) {
    std::cerr << "Error opening file: " << "/home/giordano/Desktop/UKF_ICP/lid_measures_times.txt" << std::endl;
    return;
  }
  f_lid << "timestamp   presence(yes/no) " << "\n";
  f_lid << std::fixed << std::setprecision(9);

  while (reader.has_next()) {
    auto bag_message = reader.read_next();
    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

    if (bag_message->topic_name == "/os_cloud_node/imu" ) { // "/alphasense_driver_ros/imu"  

      sensor_msgs::msg::Imu imu_msg;
      imu_serialization.deserialize_message(&serialized_msg, &imu_msg);
      
      double t = rclcpp::Time(imu_msg.header.stamp).nanoseconds() * 1e-9 - 1.62513e9;

      Eigen::Vector3d acc(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z);
      Eigen::Vector3d gyro(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);

      imu_inputs_.emplace_back(acc.x(), acc.y(), acc.z(), gyro.x(), gyro.y(), gyro.z());
      imu_dts_.push_back(t);
      ++imu;
    }
    else if (bag_message->topic_name == "/os_cloud_node/points") {

      sensor_msgs::msg::PointCloud2 cloud_msg;
      lidar_serialization.deserialize_message(&serialized_msg, &cloud_msg);
      
      double t = rclcpp::Time(cloud_msg.header.stamp).nanoseconds() * 1e-9 - 1.62513e9;

      PointCloud::Ptr cloud(new PointCloud);
      pcl::fromROSMsg(cloud_msg, *cloud);

      if (lid == 0){
        std::cout << "--- tempo misura lidar: " << t << "\n";
        for (auto &f : cloud_msg.fields){
          std::cout << f.name << std::endl;
        }
      }

      ContainerType cloud_vec;
      cloud_vec.reserve(cloud->size());
      for (const auto& p : cloud->points) {
        cloud_vec.emplace_back(p.x, p.y, p.z);
      }
      
      lidar_inputs_.push_back({t, cloud_vec});
      ++lid;
    }
  }
  /*  STAMPA DELLE SEQUENZE IMU/LIDAR
  int i=0, k=0;
  for (i=0; i<lid; ++i){
    while(imu_dts_[k]<lidar_inputs_[i].timestamp){
      const int val = (k%2==0) ? 1 : 2;
      f_imu << imu_dts_[k] << " " << val << "\n"; 
      f_lid << imu_dts_[k] << " 0 " << "\n";
      ++k;
    }

    f_imu << lidar_inputs_[i].timestamp << " 0 " << "\n";
    f_lid << lidar_inputs_[i].timestamp << " 10 " << "\n";
    
  }
  std::cout << " measures saved \n \n";  */
}

/*
void read_states(IntegrationSteps& dts_, ImuStatesExt& states, std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Errore apertura file: " << filename << std::endl;
    return;
  }
  std::string line;
  int i = 0;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(ss, cell, ',')) values.push_back(std::stod(cell));
    if (values.size() >= 8){
      // gt:  dt, px, py, pz, qw, qx, qy, qz;
      Eigen::Quaterniond quat(values[4], values[5], values[6], values[7]);
      quat.normalize();
      const Eigen::Matrix3d R       = quat.toRotationMatrix();
      const Eigen::Vector3d v       = Eigen::Vector3d::Zero();
      const Eigen::Vector3d p       = Eigen::Vector3d(values[1], values[2], values[3]);
      const Eigen::Vector3d b_gyro  = Eigen::Vector3d::Zero();
      const Eigen::Vector3d b_acc   = Eigen::Vector3d::Zero();

      NavStateExtended state;
      state.rotation() = R;
      state.velocity() = v;
      state.position() = p;
      state.biasAcc() = b_acc;
      state.biasGyro() = b_gyro;

      states.push_back(state);
      double timestamp = values[0] - 1.62513e+09;
      dts_.push_back(timestamp);
    }
    else{
      std::cout << "line " << i << " not inizialized: " << line << std::endl;
    }
    ++i;
  }
  states.shrink_to_fit();
  dts_.shrink_to_fit();
  std::cout << "Timestamp[0] for GT: " << dts_[0] << "  timestamp[N-1] for GT: " << dts_[i-1] << std::endl;
}*/

struct PreintSegment {
  Eigen::Matrix3d delta_R;   // ΔR_ij   (rotazione preintegrata)
  Eigen::Vector3d delta_v;   // Δv_ij   (velocità preintegrata, frame body i)
  Eigen::Vector3d delta_p;   // Δp_ij   (posizione preintegrata, frame body i)
  double          dt_total;  // Δt_ij   (durata totale del segmento)
  Eigen::Matrix3d J_v_ba;    // ∂Δv/∂b_a
  Eigen::Matrix3d J_p_ba;    // ∂Δp/∂b_a
  Eigen::Matrix3d J_v_bg;    // ∂Δv/∂b_g
  Eigen::Matrix3d J_p_bg;    // ∂Δp/∂b_g
};

static PreintSegment preintegrate_segment( int imu_from, int imu_to, const ImuInputs& inputs, 
    const std::vector<double>& dts_imu, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba){

  PreintSegment seg;
  seg.delta_R  = Eigen::Matrix3d::Identity();
  seg.delta_v  = Eigen::Vector3d::Zero();
  seg.delta_p  = Eigen::Vector3d::Zero();
  seg.dt_total = 0.0;
  seg.J_v_ba   = Eigen::Matrix3d::Zero();
  seg.J_p_ba   = Eigen::Matrix3d::Zero();
  seg.J_v_bg   = Eigen::Matrix3d::Zero();
  seg.J_p_bg   = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d J_R_bg = Eigen::Matrix3d::Zero(); // ∂ΔR/∂b_g (running)
  for (int k = imu_from; k < imu_to && k + 1 < (int)dts_imu.size(); ++k){
    const double dt = dts_imu[k+1] - dts_imu[k];
    if (dt <= 0.0) continue;

    const Eigen::Vector3d a = inputs[k].head<3>() - ba;   // accel corretto
    const Eigen::Vector3d w = inputs[k].tail<3>() - bg;   // gyro corretto

    const Eigen::Vector3d  w_dt  = w * dt;
    const Eigen::Matrix3d  dR_k  = lie_algebra::SO3Exp(w_dt);                
    const Eigen::Matrix3d  Jr_k  = lie_algebra::SO3_rightJacobian(w_dt);

    // --- Aggiornamento Jacobiani PRIMA di aggiornare i delta ---
    // (l'ordine conta: J_p dipende da J_v al passo precedente)
    seg.J_p_ba = seg.J_p_ba + seg.J_v_ba * dt + 0.5 * seg.delta_R * dt * dt;                      // ∂Δp/∂ba += ...
    seg.J_p_bg = seg.J_p_bg + seg.J_v_bg * dt - 0.5 * seg.delta_R * lie_algebra::skew(a) * J_R_bg * dt * dt;   // ∂Δp/∂bg 
    seg.J_v_ba = seg.J_v_ba + seg.delta_R * dt;                                                   // ∂Δv/∂ba += R_ik * dt
    seg.J_v_bg = seg.J_v_bg - seg.delta_R * lie_algebra::skew(a) * J_R_bg * dt;                            // ∂Δv/∂bg

    // J_R_bg deve aggiornarsi DOPO J_v/J_p che lo usano
    J_R_bg = dR_k.transpose() * J_R_bg - Jr_k * dt;                                               // ∂ΔR/∂bg

    // --- Aggiornamento delta su manifold ---
    seg.delta_p  = seg.delta_p + seg.delta_v * dt + 0.5 * seg.delta_R * a * dt * dt;
    seg.delta_v  = seg.delta_v + seg.delta_R * a * dt;
    seg.delta_R  = seg.delta_R * dR_k;

    seg.dt_total += dt;
  }
  return seg;
}

void init_lidar_imu(
          Eigen::Matrix<double,15,15>& P0,
          Eigen::Matrix3d&   R0,
          Eigen::Vector3d&   pos0,
          Eigen::Vector3d&   vel0,
          Eigen::Vector3d&   ba0,
          Eigen::Vector3d&   bg0,
          Eigen::Vector3d&   g_vec,
          Eigen::Matrix3d&   R_LI,
          Eigen::Vector3d&   t_LI,
          int                INIT_STEPS,   // numero di frames LIDAR usati per l'inizializzazione
          const std::vector<double>& dts_imu,
          Vector4d           imu_noise_std,
          LidarInputs&       lidar_measures,
          ImuInputs&         imu_inputs,
          Pipeline&          pipeline,
          int&               last_lidar_IDX,
          int&               last_imu_IDX){
  std::cout << "\n========== [INIT LIDAR+IMU] ==========\n";

  // intervallo comune
  double t_start = std::max(dts_imu.front(), lidar_measures.front().timestamp);
  double t_end   = std::min(dts_imu.back(),   lidar_measures.back().timestamp);

  if (t_end <= t_start) {
    std::cerr << "Nessuna sovrapposizione temporale IMU-LiDAR.\n";
    return;
  }
  std::cout << "[SYNC] intervallo comune: " << t_start << " -> " << t_end << "\n";

  // PRIMO INDICE LIDAR VALIDO
  int lidar_start_idx = 0;
  while (lidar_start_idx < (int)lidar_measures.size() && lidar_measures[lidar_start_idx].timestamp < t_start) {
    ++lidar_start_idx;
  }
  if (lidar_start_idx >= (int)lidar_measures.size()) {
    std::cerr << "Nessun frame LiDAR valido.\n";
    return;
  }

  // ACQUISIZIONE POSE DALLA FINESTRA LIDAR
  std::vector<Eigen::Isometry3d> lidar_poses;
  std::vector<double> lidar_times;
  int last_lidar_idx = lidar_start_idx;
  while (last_lidar_idx < (int)lidar_measures.size() && 
         lidar_measures[last_lidar_idx].timestamp <= t_end && 
        (int)lidar_poses.size() < INIT_STEPS) {

    const auto& meas = lidar_measures[last_lidar_idx];
    pipeline.compute(meas.timestamp, meas.cloud);
    Eigen::Isometry3d pose;
    pose.matrix() = pipeline.currentPose();

    lidar_poses.push_back(pose);
    lidar_times.push_back(meas.timestamp);
    ++last_lidar_idx;
  }
  last_lidar_IDX = last_lidar_idx;
  int N_INIT =  lidar_poses.size();   // numero di frames lidar usati effettivamente

  if (N_INIT < 3) {
    std::cerr << "Troppi pochi dati LiDAR dopo sync.\n";
    return;
  }
  std::cout << "[SYNC] frames LiDAR usati: " << N_INIT << "\n";
  std::cout << "[SYNC] t0=" << lidar_times.front() << "  tN=" << lidar_times.back() << "\n";

  // TROVA PRIMO INDICE IMU CORRISPONDENTE
  int imu_start_idx = 0;
  while (imu_start_idx < (int)dts_imu.size() && dts_imu[imu_start_idx] < t_start) { ++imu_start_idx; }

  if (imu_start_idx >= (int)dts_imu.size()) {
    std::cerr << "Nessuna IMU valida dopo sync.\n";
    return;
  }
  std::cout << "[SYNC] primo indice IMU: " << imu_start_idx << "\n";

  // PREINTEGRAZIONE MISURE IMU TRA LE POSE LIDAR
  int last_imu_idx = imu_start_idx, imu_from;
  double t1;
  std::vector<PreintSegment> segs;
  segs.reserve(N_INIT - 1);

  for (int k = 0; k < N_INIT - 1; ++k){
    //t0 = lidar_times[k];
    t1 = lidar_times[k+1];
    imu_from = last_imu_idx;
    while (last_imu_idx < (int)dts_imu.size() && dts_imu[last_imu_idx] < t1) ++last_imu_idx;

    PreintSegment seg = preintegrate_segment(
        imu_from, last_imu_idx,
        imu_inputs,
        dts_imu,
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero() );
    segs.push_back(seg);
  }
  last_imu_IDX = last_imu_idx;

  // Gyro bias
  bg0 = Eigen::Vector3d::Zero();
  for (int i = 0; i < N_INIT; ++i) {
    Eigen::Vector3d w_imu = lie_algebra::SO3Log(segs[i].delta_R) / segs[i].dt_total;
    Eigen::Matrix3d dR_lidar = lidar_poses[i].rotation().transpose() * lidar_poses[i+1].rotation();
    Eigen::Vector3d w_lidar = lie_algebra::SO3Log(dR_lidar) / (lidar_times[i+1] - lidar_times[i]);

    bg0 += (w_imu - R_LI * w_lidar);
  }
  bg0 /= N_INIT;

  // STIMA g, v, ba (sistema lineare) 
  // Ax = b
  // x = [v0, v1, ... , vn-1, g]
  const int M = N_INIT - 1; // numero di preintegrazioni
  const int STATE_SIZE = 3*N_INIT + 3; // stima velocità e g

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6*M, STATE_SIZE);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6*M);
  // Linear system in World reference frame, so the imu values are converted through the LIDAR frame into WORLD frame:
  // (w)T_i = (w)T_l * (l)T_i 

  for (int i = 0; i < M; ++i) {

    double dt = lidar_times[i+1] - lidar_times[i];

    Eigen::Matrix3d Ri = lidar_poses[i].rotation() * R_LI.transpose();
    Eigen::Vector3d pi = lidar_poses[i].translation() - Ri * t_LI;
    Eigen::Vector3d pj = lidar_poses[i+1].translation() - (lidar_poses[i+1].rotation() * R_LI.transpose()) * t_LI;

    // velocity equation: - v_i + v_j - g*dt = (w)R_I * dv_ij
    int row = 6*i;
    A.block<3,3>(row,3*i)     = -Eigen::Matrix3d::Identity();       // - v_i
    A.block<3,3>(row,3*(i+1)) =  Eigen::Matrix3d::Identity();       // + v_j
    A.block<3,3>(row,3*N_INIT)     = -dt*Eigen::Matrix3d::Identity();    // - g*dt

    b.segment<3>(row) = Ri * segs[i].delta_v;                       // (w)R_I * dv_ij

    // position equation: - v_i * dt - 0.5 * g * dt^2 = pi - pj + Ri * dp_ij
    row += 3;
    A.block<3,3>(row,3*i) = -dt*Eigen::Matrix3d::Identity();        // - v_i * dt
    A.block<3,3>(row,3*N_INIT) = -0.5*dt*dt*Eigen::Matrix3d::Identity(); // - 0.5 * g * dt^2

    b.segment<3>(row) = pi - pj + Ri * segs[i].delta_p;             // pi - pj + Ri * dp_ij
  }
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

  g_vec = x.segment<3>(3*N_INIT);
  g_vec = 9.81 * g_vec.normalized();

  vel0 = x.segment<3>(3*(N_INIT-1));

  R0 = lidar_poses[N_INIT-1].rotation() * R_LI.transpose();

  pos0 = lidar_poses[N_INIT-1].translation() - R0 * t_LI;
  
  ba0 = imu_inputs[0].head<3>() + R0.transpose()*g_vec;

  // GAUSS NEWTON REFINEMENT for g, bg0 and ba0
  // using the same equations of the linear system to determine the residuals of position and velocity
  /*for(int iter=0; iter<10; iter++) {

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(9,9);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(9);  // state = [gravity(3), bg0(3), ba0(3)]

    last_imu_idx = imu_start_idx;

    for(int i=0; i<M; ++i) {

      t0 = lidar_times[i];
      t1 = lidar_times[i+1];
      imu_from = last_imu_idx;
      while (last_imu_idx < (int)dts_imu.size() && dts_imu[last_imu_idx] < t1) ++last_imu_idx;

      PreintSegment seg = preintegrate_segment(t0,t1,imu_inputs,dts_imu,bg0,ba0);
      double dt = segs[i].dt_total;

      Eigen::Matrix3d Ri = lidar_poses[i].rotation() * R_LI.transpose();
      Eigen::Vector3d pi = lidar_poses[i].translation() - Ri*t_LI;
      Eigen::Vector3d pj = lidar_poses[i+1].translation() - (lidar_poses[i+1].rotation() *R_LI.transpose())*t_LI;

      Eigen::Vector3d vi = x.segment<3>(3*i);
      Eigen::Vector3d vj = x.segment<3>(3*(i+1));

      Eigen::Vector3d rv = Ri.transpose() * (vj-vi-g_vec*dt) - segs[i].delta_v;
      Eigen::Vector3d rp = Ri.transpose() * (pj-pi-vi*dt-0.5*g_vec*dt*dt) - segs[i].delta_p;

      Eigen::Matrix<double,6,9> J;
      J.setZero();

      J.block<3,3>(0,0) = -Ri.transpose()*dt;
      J.block<3,3>(3,0) = -0.5*Ri.transpose()*dt*dt;
      J.block<3,3>(0,3) = -segs[i].J_v_bg;
      J.block<3,3>(3,3) = -segs[i].J_p_bg;
      J.block<3,3>(0,6) = -segs[i].J_v_ba;
      J.block<3,3>(3,6) = -segs[i].J_p_ba;

      Eigen::Matrix<double,6,1> r;
      r << rv, rp;

      H += J.transpose()*J;
      g += J.transpose()*r;
    }

    Eigen::VectorXd dx = H.ldlt().solve(-g);

    g_vec += dx.segment<3>(0);
    bg0   += dx.segment<3>(3);
    ba0   += dx.segment<3>(6);

    g_vec = 9.81*g_vec.normalized();

    if(dx.norm() < 1e-5) break;
  }*/

  // COVARIANCE
  //P0.setIdentity();
  //P0 *= 0.1;

  P0.setZero();
  P0(0,0) = 1e-3;
  P0(1,1) = 1e-3;
  P0(2,2) = 1;
  P0.block<3,3>(3,3)   = 0.1 * Eigen::Matrix3d::Identity();
  P0.block<3,3>(6,6)   = 0.1 * Eigen::Matrix3d::Identity();
  P0.block<3,3>(9,9)   = pow(imu_noise_std(2),2) * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = pow(imu_noise_std(3),2) * Eigen::Matrix3d::Identity();

  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [ ";
    for (int col = 0; col < 3; ++col) std::cout << std::setw(9) << R0(row,col) << " ";
    std::cout << "]\n";
  }
  std::cout << " R_LI:\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [ ";
    for (int col = 0; col < 3; ++col) std::cout << std::setw(9) << R_LI(row,col) << " ";
    std::cout << "]\n";
  }
  std::cout << " pos0=" << pos0.transpose() << "\n";
  std::cout << " vel0=" << vel0.transpose() << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose() << "\n";
  std::cout << " P0 diag=" << P0.diagonal().transpose() << "\n";
  std::cout << " g_vec=" << g_vec.transpose() << "\n";
  std::cout << " IMU   used in init: " << last_imu_IDX - imu_start_idx << " last IMU    index: " << last_imu_IDX << "\n";
  std::cout << " LIDAR used in init: " << last_lidar_IDX - lidar_start_idx << " last LIDAR index: " << last_lidar_IDX << "\n";

  std::cout << "========== END INIT2 ==========\n\n";
}

void init_lidar_imu_3(
          Eigen::Matrix<double,24,24>& P0,
          Eigen::Matrix3d&   R0,
          Eigen::Vector3d&   pos0,
          Eigen::Vector3d&   vel0,
          Eigen::Vector3d&   ba0,
          Eigen::Vector3d&   bg0,
          Eigen::Vector3d&   g_vec,
          Eigen::Matrix3d&   R_LI,
          Eigen::Vector3d&   t_LI,
          int                INIT_STEPS,   // numero di frames LIDAR usati per l'inizializzazione
          const std::vector<double>& dts_imu,
          Vector4d           imu_noise_std,
          LidarInputs&       lidar_measures,
          ImuInputs&         imu_inputs,
          Pipeline&          pipeline,
          int&               last_imu_IDX,
          int&               last_lidar_IDX){

  std::cout << "\n=== FASTLIO STYLE INIT ===\n";

  // SYNCHRONIZATION MEASURES LIDAR IMU
  if (dts_imu.empty() || lidar_measures.empty()) {
    std::cerr << "IMU o LiDAR vuoti.\n";
    return;
  }
  // intervallo comune
  double t_start = std::max(dts_imu.front(), lidar_measures.front().timestamp);
  double t_end   = std::min(dts_imu.back(),   lidar_measures.back().timestamp);

  std::cout << "[SYNC] intervallo comune: " << t_start << " -> " << t_end << "\n";

  // PRIMO INDICE LIDAR VALIDO
  int lidar_start_idx = 0;
  while (lidar_start_idx < (int)lidar_measures.size() && lidar_measures[lidar_start_idx].timestamp < t_start) {
    ++lidar_start_idx;
  }
  std::cout << "[SYNC] primo indice LIDAR: " << lidar_start_idx << "\n";
  last_lidar_IDX = lidar_start_idx;

  // TROVA PRIMO INDICE IMU CORRISPONDENTE
  int imu_start_idx = 0;
  while (imu_start_idx < (int)dts_imu.size() && dts_imu[imu_start_idx] < lidar_measures[lidar_start_idx].timestamp) { 
    ++imu_start_idx; 
  }
  std::cout << "[SYNC] primo indice IMU: " << imu_start_idx << "\n";
  last_imu_IDX = imu_start_idx;
  // la prima misura IMU valida per la sincronizzazione con le misure LIDAR è "imu_start_idx"
  // quindi le misure IMU antecedenti possono essere utilizzate per la stima di:
  // - g
  // - bias gyro
  // - cov acc
  // - cov gyro
  Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_gyr = Eigen::Vector3d::Zero();
  Eigen::Vector3d cov_acc  = Eigen::Vector3d::Zero();
  Eigen::Vector3d cov_gyr  = Eigen::Vector3d::Zero();
  Eigen::Vector3d curr_acc, curr_gyr;

  for(int temp_idx = 0; temp_idx < imu_start_idx; ++temp_idx){
    curr_acc = imu_inputs[temp_idx].head<3>();
    curr_gyr = imu_inputs[temp_idx].tail<3>();
    std::cout << " --- timestamp =" << dts_imu[temp_idx] << " ";
    std::cout << " --- curr_acc =" << curr_acc.transpose() << " ";
    std::cout << " --- curr_gyr =" << curr_gyr.transpose() << "\n";

    mean_acc += (curr_acc - mean_acc)/(temp_idx+1);
    mean_gyr += (curr_gyr - mean_gyr)/(temp_idx+1);

    cov_acc = cov_acc*(temp_idx)/(temp_idx+1) + (curr_acc-mean_acc).cwiseProduct(curr_acc-mean_acc)*(temp_idx)/((temp_idx+1)*(temp_idx+1));
    cov_gyr = cov_gyr*(temp_idx)/(temp_idx+1) + (curr_gyr-mean_gyr).cwiseProduct(curr_gyr-mean_gyr)*(temp_idx)/((temp_idx+1)*(temp_idx+1));
  }
  std::cout << "\n --- mean_acc =" << mean_acc.transpose() << "\n";
  std::cout << " --- mean_acc.normalized() =" << mean_acc.normalized().transpose() << "\n";

  imu_noise_std(0) = cov_acc.norm();
  imu_noise_std(1) = cov_gyr.norm();

  std::cout << "\n --- cov_acc =" << cov_acc.transpose() << " --- cov_acc.norm() =" << cov_acc.norm();
  std::cout << "\n --- cov_gyr =" << cov_gyr.transpose() << " --- cov_gyr.norm() =" << cov_gyr.norm() << "\n";

  // gravity
  g_vec = -9.81 * mean_acc.normalized();
  std::cout << "\n --- g_vec =" << g_vec.transpose() << "\n";

  // gyro bias
  bg0 = mean_gyr;
  std::cout << " --- bg0 =" << bg0.transpose() << "\n";

  // COVARIANCE
  P0.setZero();
  P0.block<3,3>(0,0)   = 0.01 * Eigen::Matrix3d::Identity();                      // Rot
  P0.block<3,3>(3,3)   = 0.01 * Eigen::Matrix3d::Identity();                      // pos
  P0.block<3,3>(6,6)   = 0.01 * Eigen::Matrix3d::Identity();                      // vel
  P0.block<3,3>(9,9)   = 0.01 * Eigen::Matrix3d::Identity();//pow(imu_noise_std(3),2) * Eigen::Matrix3d::Identity();  // bias_gyro
  P0.block<3,3>(12,12) = 0.01 * Eigen::Matrix3d::Identity();//pow(imu_noise_std(2),2) * Eigen::Matrix3d::Identity();  // bias_acc
  P0.block<3,3>(15,15) = 0.0001 * Eigen::Matrix3d::Identity();                  // g_vec
  P0.block<3,3>(18,18) = 0.00001 * Eigen::Matrix3d::Identity();                  // R_li
  P0.block<3,3>(21,21) = 0.00001 * Eigen::Matrix3d::Identity();                  // t_li
/*----------------------------------------------------------------------------------------------
  int last_lidar_idx = lidar_start_idx + INIT_STEPS;
  double t_end_init = lidar_measures[last_lidar_idx].timestamp;
  std::cout << "[SYNC] timestamp finale per init: " << t_end_init << "\n";
  last_lidar_IDX = last_lidar_idx;

  
  last_imu_IDX = last_imu_idx;

  // costruzione R0 da gravity
  Eigen::Vector3d g_world(0,0,-1);
  Eigen::Vector3d g_imu = mean_acc.normalized();
  Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(g_world, g_imu);
  R0 = q.toRotationMatrix();  // R_IMU in WORLD frame

  // Ruoto il LIDAR frame per allinearlo al WORLD frame, 
  // così da rendere il vettore gravità stimato coerente coi calcoli da IMU
  Eigen::Isometry3d T;
  T.linear() = R0 * R_LI;  // R_LIDAR in World = R_IMU in World * R_LIDAR in Imu
  pipeline.initialize_from_RF(lidar_measures[0].timestamp, &lidar_measures[0].cloud, T);

  // posizione iniziale da lidar
  int k = 1;
  while(lidar_measures[k].timestamp < t_end_init){
    pipeline.compute(lidar_measures[k].timestamp, lidar_measures[k].cloud);
    ++k;
  }

  Eigen::Isometry3d pose;
  pose.matrix() = pipeline.currentPose();

  Eigen::Matrix3d Rwl = pose.rotation();
  Eigen::Vector3d twl = pose.translation();

  pos0 = twl - R0 * t_LI;

  // velocità iniziale
  vel0.setZero();

  // acc bias
  ba0 = mean_acc + R0.transpose()*g_vec;

  // covarianza
  P0.setIdentity();
  //P0 *= 0.01;

  std::cout << "R0=\n" << R0 << "\n";
  std::cout << "g=" << g_vec.transpose() << "\n";
  std::cout << "bg=" << bg0.transpose() << "\n";
  std::cout << "ba=" << ba0.transpose() << "\n";

----------------------------------------------------------------------------------------------*/
}

int main(int argc, char** argv){
  if (argc < 3) {
    std::cout << "usage: this-executable path-to-output trajectory-type" << std::endl;
    std::cout << "example: ./example_UKF_NEWCO /home/giordano/Desktop/UKF_ICP 1" << std::endl;
    return 1;
  }
  // STARTING TIME
  auto t0 = std::chrono::high_resolution_clock::now();

  int Traj = 1;
    if (argc >= 3) {
      try {
        Traj = std::stoi(argv[2]);
      }
      catch (const std::exception&) {
        std::cerr << "trajectory-type must be an integer (1-4):\n" 
                  << "            1 -> quad easy\n"
                  << "            2 -> quad medium\n"
                  << "            3 -> quad hard\n"
                  << "            4 -> stairs\n" << std::endl;
        return 1;
      }
    }
    std::string dataset_filename;
    switch (Traj) {
      case 1:
        dataset_filename = data_folder + "/../../NewerCollegeDataset/Collection_1/quad_easy/ros2";
        break;
      case 2:
        dataset_filename = data_folder + "/../../NewerCollegeDataset/Collection_1/quad_medium/ros2";
        break;
      case 3:
        dataset_filename = data_folder + "/../../NewerCollegeDataset/Collection_1/quad_hard/ros2";
        break;
      case 4:
        dataset_filename = data_folder + "/../../NewerCollegeDataset/Collection_1/stairs/ros2";
        break;
      default:
        std::cerr << "Invalid trajectory type.\n"
                  << "1 -> quad_easy\n"
                  << "2 -> quad_medium\n"
                  << "3 -> quad_hard\n"
                  << "4 -> stairs\n";
        return 1;
    }

  Pipeline pipeline(
    10.0,   // sensor_hz
    false,  // deskew
    0.2,    // b_max
    0.1,    // rho_ker
    0.8,    // p_th
    0.1,    // b_min
    0.02,   // b_ratio
    4,      // num_keyframes
    4,      // num_cores
    false   // realtime
  );

  Vector4d imu_noise_std;
  imu_noise_std << 1.86e-03,   // sigma_noise_accel - 8e-2
                   1.87e-04,   // sigma_noise_gyro  - 4e-3
                   4.33e-04,   // sigma_walk_accel  - 4e-5
                   2.66e-05;   // sigma_walk_gyro   - 2e-6
  
  int INIT_STEPS  = 16;   // numero di lidar frames per l'inizializzazione

  //ImuStatesExt gt_states, gt_interp;
  ImuInputs imu_inputs;
  LidarInputs lidar_measures;
  std::vector<int> imu_indices;
  IntegrationSteps dts_imu, dts_states; 

  //std::string gt_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/ground-truth/tum_format/gt-nc-quad-medium.csv";

  read_ros2_bag_imu(dts_imu, imu_inputs, lidar_measures, dataset_filename);
  //read_states(dts_states, gt_states, gt_filename);

  // trasformazione T_li ricavata dal dataset:  LIDAR -> IMU  (esprime le coordinate del frame LIDAR rispetto l'IMU frame)
  // p_imu  =  T_li * p_lidar  =  R_li * p_lidar + t_li
  
  /*Eigen::Matrix3d R_LI;
  R_LI << 1,0,0,
          0,-1,0,
          0,0,-1;
  Eigen::Vector3d t_LI;
  t_LI << -0.065, -0.00855, -0.0336;*/

  Eigen::Matrix3d R_LI;
  R_LI << 1,0,0,
          0,1,0,
          0,0,1;
  Eigen::Vector3d t_LI;
  t_LI << 0.0, 0.0, 0.0;

  std::cout << "=== POST READING FUNCTIONS ===\n";
  std::cout << "dts_imu.size()        = " << dts_imu.size()        << "\n";
  std::cout << "imu_inputs.size()     = " << imu_inputs.size()     << "\n";
  std::cout << "lidar_measures.size() = " << lidar_measures.size() << "\n";
  //std::cout << "gt_states.size()      = " << gt_states.size()      << "\n";
  //std::cout << "dts_states.size()     = " << dts_states.size()     << "\n";
  
  // FILTER INIZIALIZATION
  NavStateLidar ukf_state;
  NavStateExtended ukf_state_ext;

  Matrix12d Q = Matrix12d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);// noise_acc
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);// noise_gyro
  Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);// bias_acc
  Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);// bias_gyro
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);
  const Matrix6d Q_bias = Q.block<6,6>(6,6);
  const Matrix6d Q_inputs = Q.block<6,6>(0,0);

  Eigen::Matrix<double,24,24> P0;
  Eigen::Matrix3d R0;
  Eigen::Vector3d ba0, bg0, pos0, vel0, g_vec;
  P0.setIdentity();
  R0.setIdentity();
  ba0 = bg0 = pos0 = vel0 = Eigen::Vector3d(0, 0, 0);
  g_vec = Eigen::Vector3d(0, 0, -9.81);
  int start_imu_idx, start_lidar_idx;
  
  // using first measures of IMU for initialization
  init_lidar_imu_3(P0, R0, pos0, vel0, ba0, bg0, g_vec, R_LI, t_LI, INIT_STEPS, dts_imu, 
                 imu_noise_std, lidar_measures, imu_inputs, pipeline, start_imu_idx, start_lidar_idx);

  ukf_state.rotation()   = R0;
  ukf_state.position()   = pos0;
  ukf_state.velocity()   = vel0;
  ukf_state.biasGyro()   = bg0;
  ukf_state.biasAcc()    = ba0;
  ukf_state.gravity()    = g_vec;
  ukf_state.R_li()       = R_LI;
  ukf_state.t_li()       = t_LI;
  ukf_state.covariance() = P0;

  ukf_state_ext.rotation()   = R0;
  ukf_state_ext.position()   = pos0;
  ukf_state_ext.velocity()   = vel0;
  ukf_state_ext.biasGyro()   = bg0;
  ukf_state_ext.biasAcc()    = ba0;
  ukf_state_ext.gravity()    = g_vec;
  ukf_state_ext.covariance() = P0.block<15,15>(0, 0);

  // ------- UKF filter
  UKFImuLidar ukf_lidar;
  ukf_lidar.setWeights(ukf_state.stateDim(), 12, alpha);
  UKFImuExtendedOdom ukf_ext;
  ukf_ext.setWeights(ukf_state_ext.stateDim(), 6, alpha);

  // -------- initializing estimates
  int N = imu_inputs.size() - start_imu_idx;  

  ImuLidarStates estimates_lidar(N);
  ImuStatesExt estimates_extended(N);
  estimates_lidar[0]       = ukf_state;
  estimates_extended[0]    = ukf_state_ext;

  OdomObs obs;
  //Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
  //offset.linear() = R_LI;
  //offset.translation() = t_LI;
  //obs.setOffset(offset);

  std::cout<<"\nUKF-LIDAR: state dimension: "<< ukf_state.stateDim() <<" and observation dimension: "<< obs.obsDim() <<"\n";
  std::cout<<"UKF-EXTENDED: state dimension: "<< ukf_state_ext.stateDim() <<" and observation dimension: "<< obs.obsDim() <<"\n";

  Eigen::MatrixXd R, H;
  double DT;
  double lidar_idx = start_lidar_idx;
  Eigen::Isometry3d Z;
  Vector6d z_tang;
  //pipeline.compute(lidar_measures[lidar_idx].timestamp, lidar_measures[lidar_idx].cloud);
  int l=0;

  std::cout << "=== STARTING FILTER LOOP ===\n";

  // START FILTERING MEASURES
  for (int k = start_imu_idx; k < N; ++k) {

    if (k%1000 == 0){
      if (l%4 == 0){
        std::cout << "\n--- iteration: " << k << " ---";
      }
      else{
        std::cout << "   iteration: " << k << " ---";
      }
      l++;
    } 
    
    double t0 = dts_imu[k-1];
    double t1 = dts_imu[k];
    DT = t1 - t0;
    if (DT <= 0) continue;

    ukf_lidar.propagate(ukf_state, imu_inputs.at(k-1), Q, DT);
    ukf_ext.propagate(ukf_state_ext, imu_inputs.at(k-1), Q_inputs, DT);

    ukf_state_ext.covariance().block<6,6>(9,9) += Q_bias * DT * DT;

    if (lidar_idx < lidar_measures.size()-1 && lidar_measures[lidar_idx].timestamp <= t1) {

      const auto& meas = lidar_measures[lidar_idx+1];

      // passo la cloudpoint a MAD-ICP
      pipeline.compute(meas.timestamp, meas.cloud);

      // ottengo la trasformazione rispetto alla mappa locale MAD-ICP 
      // la converto nello spazio tangente solamente per il plotting
      Z = pipeline.currentPose();
      //z_tang.head(3) = t_LI + R_LI*(Z.translation() + Z.rotation()*(-R_LI.transpose()*t_LI)); 
      //z_tang.tail(3) = lie_algebra::SO3Log(R_LI * Z.rotation() * R_LI.transpose());
      z_tang.head(3) = Z.translation();
      z_tang.tail(3) = lie_algebra::SO3Log(Z.rotation());

      // ricavo le informazioni da mad-icp per calcolare la matrice di covarianza
      H = pipeline.getHinfo();
      H = H + 1e-6 * Matrix6d::Identity();   // termine correttivo per evitare errori numerici
      R = H.inverse();

      if (!R.allFinite()) {
        std::cout << "NaN estimate at k=" << k << " and at lidar index =" << lidar_idx << "\n";
        break;
      }

      ukf_lidar.write_log("162513");
      ukf_lidar.write_log(t1);
      ukf_lidar.write_pipe("162513");
      ukf_lidar.write_pipe(t1);
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(z_tang(0));
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(z_tang(1));
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(z_tang(2));
      Eigen::Quaterniond q(Z.rotation());
      q.normalize();
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(q.x());
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(q.y());
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(q.z());
      ukf_lidar.write_pipe(" ");
      ukf_lidar.write_pipe(q.w());
      ukf_lidar.write_pipe("\n");

      obs.meas() = Z;

      ukf_ext.update(ukf_state_ext, obs, R);
      ukf_lidar.update(ukf_state, obs, R);

      lidar_idx++;
    } 
    if (lidar_idx == lidar_measures.size()) {
      std::cout << "Lidar measures finished at k=" << k << "\n";
      break;
    }
    if (!ukf_state.position().allFinite()) {
      std::cout << "NaN estimate for ukf_lidar at k=" << k << "\n";
      break;
    }

    estimates_lidar.at(k - start_imu_idx + 1) = ukf_state;
    estimates_extended.at(k - start_imu_idx + 1) = ukf_state_ext;
  }
  std::cout << "\n\n=== ENDING FILTER LOOP ===\n";


  //std::cout << "Matrice di covarianza P post-filtraggio:\n" << ukf_state.covariance() << std::endl;
  
  // output file
  auto write_traj = [&](const std::string& fname, const ImuLidarStates& states, const IntegrationSteps& dts, const Vector15d& vec, int prec){
    std::ofstream f(fname);
    if (!f.is_open()) {
      std::cerr << "Error opening file: " << fname << std::endl;
      return;
    }
    f << std::fixed << std::setprecision(prec);
    size_t N = std::min(states.size(), dts.size());
    for (size_t i = 0; i < N; ++i) {
      const auto& s = states[i];
      Eigen::Quaterniond q(s.rotation());
      q.normalize();
      f << "162513" << dts[i] << " ";
      if (vec[0]==1)  f << s.position().x() << " " << s.position().y() << " " << s.position().z() << " ";          // pos
      if (vec[1]==1)  f << q.x() << " " << q.y() << " " << q.z() << " " << q.w();                                  // rot
      if (vec[2]==1)  f << " " << s.velocity().x() << " " << s.velocity().y() << " " << s.velocity().z() << " ";   // vel
      if (vec[3]==1)  f << " " << s.biasGyro().x() << " " << s.biasGyro().y() << " " << s.biasGyro().z() << " ";   // bias_gyro
      if (vec[4]==1)  f << " " << s.biasAcc().x() << " " << s.biasAcc().y() << " " << s.biasAcc().z() << " ";      // bias_acc
      if (vec[5]==1)  f << " " << s.gravity().x() << " " << s.gravity().y() << " " << s.gravity().z() << " ";      // gravity

      if (vec[6]==1)  f << " " << s.covariance().diagonal().segment<3>(0).transpose() << " ";                      // cov rot
      if (vec[7]==1)  f << " " << s.covariance().diagonal().segment<3>(3).transpose() << " ";                      // cov pos
      if (vec[8]==1)  f << " " << s.covariance().diagonal().segment<3>(6).transpose() << " ";                      // cov vel
      if (vec[9]==1)  f << " " << s.covariance().diagonal().segment<3>(9).transpose() << " ";                      // cov b_gyro
      if (vec[10]==1) f << " " << s.covariance().diagonal().segment<3>(12).transpose() << " ";                     // cov b_acc
      if (vec[11]==1) f << " " << s.covariance().diagonal().segment<3>(15).transpose() << " ";                     // cov grav
      if (vec[12]==1) f << " " << s.covariance().diagonal().segment<3>(18).transpose() << " ";                     // cov R_li
      if (vec[12]==1) f << " " << s.covariance().diagonal().segment<3>(21).transpose() << " ";                     // cov t_li
      f << "\n";
    }
    std::cout << " wrote file: " << fname << " | len: " << N << std::endl;
  };

  auto write_traj_ext = [&](const std::string& fname, const ImuStatesExt& states, const IntegrationSteps& dts, const Vector15d& vec, int prec){
    std::ofstream f(fname);
    if (!f.is_open()) {
      std::cerr << "Error opening file: " << fname << std::endl;
      return;
    }
    f << std::fixed << std::setprecision(prec);
    size_t N = std::min(states.size(), dts.size());
    for (size_t i = 0; i < N; ++i) {
      const auto& s = states[i];
      Eigen::Quaterniond q(s.rotation());
      q.normalize();
      f << "162513" << dts[i] << " ";
      if (vec[0]==1)  f << s.position().x() << " " << s.position().y() << " " << s.position().z() << " ";          // pos
      if (vec[1]==1)  f << q.x() << " " << q.y() << " " << q.z() << " " << q.w();                                  // rot
      if (vec[2]==1)  f << " " << s.velocity().x() << " " << s.velocity().y() << " " << s.velocity().z() << " ";   // vel
      if (vec[3]==1)  f << " " << s.biasGyro().x() << " " << s.biasGyro().y() << " " << s.biasGyro().z() << " ";   // bias_gyro
      if (vec[4]==1)  f << " " << s.biasAcc().x() << " " << s.biasAcc().y() << " " << s.biasAcc().z() << " ";      // bias_acc
      if (vec[5]==1)  f << " " << s.gravity().x() << " " << s.gravity().y() << " " << s.gravity().z() << " ";      // gravity

      if (vec[6]==1)  f << " " << s.covariance().diagonal().segment<3>(0).transpose() << " ";                      // cov rot
      if (vec[7]==1)  f << " " << s.covariance().diagonal().segment<3>(3).transpose() << " ";                      // cov pos
      if (vec[8]==1)  f << " " << s.covariance().diagonal().segment<3>(6).transpose() << " ";                      // cov vel
      if (vec[9]==1)  f << " " << s.covariance().diagonal().segment<3>(9).transpose() << " ";                      // cov b_gyro
      if (vec[10]==1) f << " " << s.covariance().diagonal().segment<3>(12).transpose() << " ";                     // cov b_acc
      f << "\n";
    }
    std::cout << " wrote file: " << fname << " | len: " << N << std::endl;
  };

  auto write_traj_csv = [&](const std::string& fname, const ImuLidarStates& states, const IntegrationSteps& dts){
    std::ofstream f(fname);
    if (!f.is_open()) {
      std::cerr << "Error opening file: " << fname << std::endl;
      return;
    }
    f << "t,roll,pitch,yaw,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz,acc_x,acc_y,acc_z\n";
    f << std::fixed << std::setprecision(6);
    size_t N = std::min(states.size(), dts.size());
    Eigen::Vector3d euler;
    for (size_t i = 0; i < N; ++i) {
      const auto& s = states[i];
      euler = rotMatToEuler(s.rotation());

      f << dts[i] << ","
        << euler[0] << "," << euler[1] << "," << euler[2] << ","
        << s.position().x() << "," << s.position().y() << "," << s.position().z() << ","
        << s.velocity().x() << "," << s.velocity().y() << "," << s.velocity().z() << ","
        << s.biasGyro().x() << "," << s.biasGyro().y() << "," << s.biasGyro().z() << ","
        << s.biasAcc().x() << "," << s.biasAcc().y() << "," << s.biasAcc().z() << ","
        << s.gravity().x() << "," << s.gravity().y() << "," << s.gravity().z() << ","
        << std::endl;
    }
    std::cout << " wrote file: " << fname << " | len: " << N << std::endl;
  };

  std::string out = std::string(argv[1]);
  Vector15d vect;
  //write_traj(out+"/gt_states.txt", gt_states, dts_states);

  vect << 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  write_traj(out+"/est_lio.txt",  estimates_lidar, dts_imu, vect, 6);

  vect << 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  write_traj_ext(out+"/est_lio_ext.txt",  estimates_extended, dts_imu, vect, 6);

  vect << 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0; 
  write_traj(out+"/covariance_r_p_v.txt",  estimates_lidar, dts_imu, vect, 9);

  vect << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0; 
  write_traj(out+"/covariance_bg_ba_g.txt",  estimates_lidar, dts_imu, vect, 9);

  vect << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1; 
  write_traj(out+"/covariance_rli_tli.txt",  estimates_lidar, dts_imu, vect, 9);

  vect << 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0; 
  write_traj(out+"/bg_ba_grav.txt",  estimates_lidar, dts_imu, vect, 9);

  write_traj_csv(out+"/state_log.csv",  estimates_lidar, dts_imu);

  // ENDING TIME
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Tempo esecuzione: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  std::cout << "job finished." << std::endl;

  return 0;
}
