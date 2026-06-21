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
#include <pcl/register_point_struct.h>
#include <pcl_conversions/pcl_conversions.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// -------------------------------------------- Utilities for usage: -----------------------------------------
//
// ./LIO-UKF-MADICP /home/giordano/Desktop/UKF_ICP 2
//
// ./this-executable path-to-output trajectory-type
//
// ------------------------------------------- Utilities for plotting: ----------------------------------------
//
//  splot "MAD-ICP_tum.txt" using 2:3:4 w l, "pipeline_pos_und.txt" using 2:3:4 w l, "est_lio_und.txt" using 2:3:4 w l, "gt_tum.txt" using 2:3:4 w l
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ouster_ros{
  struct EIGEN_ALIGN16 Point{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  // use std::uint32_t to avoid conflicting with pcl::uint32_t
  (std::uint32_t, t, t)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint8_t, ring, ring)
  (std::uint16_t, ambient, ambient)
  (std::uint32_t, range, range)
)

using namespace ukf_manifold;

const std::string data_folder(UKF_DATA_FOLDER);

using ImuStates    = std::vector<NavState>;
using ImuStatesExt = std::vector<NavStateExtended>;

using Vector4d    = Eigen::Matrix<double, 4, 1>;
using Matrix12d   = Eigen::Matrix<double, 12, 12>;


struct ImuMeasurement {
  double timestamp;
  Vector6d acc_gyro;
};
using ImuInputs = std::vector<ImuMeasurement>;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloud;
using ContainerType = std::vector<Eigen::Vector3d>;

struct LidarMeasurement {
  double timestamp;
  double end_time;
  PointCloud::Ptr cloud;
  //ContainerType cloud_undistorted;
};
using LidarInputs = std::vector<LidarMeasurement>;

void read_ros2_bag_imu(
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

  while (reader.has_next()) {
    auto bag_message = reader.read_next();
    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

    if (bag_message->topic_name == "/os_cloud_node/imu" ) { // "/alphasense_driver_ros/imu"  

      sensor_msgs::msg::Imu imu_msg;
      imu_serialization.deserialize_message(&serialized_msg, &imu_msg);
      
      double t = rclcpp::Time(imu_msg.header.stamp).nanoseconds() * 1e-9 - 1.62513e9;
      if (imu == 0){
        std::cout << "--- tempo misura imu: " << t << "\n";

      }

      Vector6d acc_gyro(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z,
                               imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);

      imu_inputs_.push_back({t, acc_gyro});
      ++imu;
    }
    else if (bag_message->topic_name == "/os_cloud_node/points") {

      sensor_msgs::msg::PointCloud2 cloud_msg;
      lidar_serialization.deserialize_message(&serialized_msg, &cloud_msg);

      double t = rclcpp::Time(cloud_msg.header.stamp).nanoseconds() * 1e-9 - 1.62513e9;

      if (lid == 0){
        std::cout << "--- tempo misura lidar: " << t << "\n";
        std::cout << "--- campi pointcloud: " << "\n";
        for (auto &f : cloud_msg.fields){
          std::cout << f.name << " / ";
        }
        std::cout<<"\n";
      }

      pcl::PointCloud<ouster_ros::Point> raw_cloud;  // OSTER CLOUDPOINT
      pcl::fromROSMsg(cloud_msg, raw_cloud);

      PointCloud::Ptr cloud(new PointCloud);         // FASTER CLOUDPOINT
      cloud->reserve(raw_cloud.size());

      double end_time = raw_cloud.points.front().t * 1e-6;
      
      for (const auto& src : raw_cloud.points) {

        PointType dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = src.intensity;
        dst.normal_x = 0.0;
        dst.normal_y = 0.0;
        dst.normal_z = 0.0;

        // OUSTER time per punto in ns ---> curvature in ms
        dst.curvature = src.t * 1e-6;
        if (dst.curvature > end_time) end_time = dst.curvature;

        cloud->push_back(dst);
      }

      LidarMeasurement lid_meas;
      lid_meas.timestamp = t;
      lid_meas.end_time  = end_time * 1e-3 + t;
      lid_meas.cloud     = cloud;
      if (lid == 0){
        std::cout << "--- tempo finale misura lidar: " << lid_meas.end_time << "\n";
      }
      
      lidar_inputs_.push_back(lid_meas);
      ++lid;
    }
  }

  std::cout << "Timestamp[0] for IMU: " << imu_inputs_[0].timestamp 
            << " / timestamp[N-1] for IMU: " << imu_inputs_[imu-1].timestamp  << std::endl;
  std::cout << "Timestamp[0] for LIDAR: " << lidar_inputs_[0].timestamp 
            << " / timestamp[N-1] for LIDAR: " << lidar_inputs_[lid-1].timestamp << std::endl;

}

struct ImuInit {
    Eigen::Vector3d mean_acc  = Eigen::Vector3d(0, 0, -1.0);
    Eigen::Vector3d mean_gyr  = Eigen::Vector3d::Zero();
    Eigen::Vector3d cov_acc   = Eigen::Vector3d(0.1, 0.1, 0.1);
    Eigen::Vector3d cov_gyr   = Eigen::Vector3d(0.1, 0.1, 0.1);
    int N = 0;             // contatore campioni (algoritmo di Welford)
    bool done = false;     // true quando init_iter_num > N_INIT
};

struct ImuPose{
    double offset_time = 0.0;
    Eigen::Vector3d pos  = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel  = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot  = Eigen::Matrix3d::Identity();
    Eigen::Vector3d acc  = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr  = Eigen::Vector3d::Zero();
};

bool imu_init_accumulate(
        const ImuInputs& imu_meas,
        ImuInit&         init_imu,
        int              N_INIT){
  if (imu_meas.empty()) return false;
  std::cout << std::fixed << std::setprecision(6);

  // Al primo MeasureGroup inizializza le medie col primo campione
  if (init_imu.N == 0) {
    init_imu.mean_acc = imu_meas.front().acc_gyro.head<3>();
    init_imu.mean_gyr = imu_meas.front().acc_gyro.tail<3>();
    init_imu.N = 1;
  }

  // Aggiornamento di Welford su tutti i campioni IMU del MeasureGroup
  for (const ImuMeasurement& imu : imu_meas) {
    // mean_new = mean_old + (x - mean_old) / N
    std::cout<<" Timestamp: "<<imu.timestamp;
    std::cout<<" cur_acc: "<<imu.acc_gyro.head<3>().transpose()<<" , mean_acc: "<<init_imu.mean_acc.transpose()
        <<", cur_gyr: "<<imu.acc_gyro.tail<3>().transpose()<<" , mean_gyr: "<<init_imu.mean_gyr.transpose()<<std::endl;

    init_imu.mean_acc += (imu.acc_gyro.head<3>() - init_imu.mean_acc) / init_imu.N;
    init_imu.mean_gyr += (imu.acc_gyro.tail<3>() - init_imu.mean_gyr) / init_imu.N;

    init_imu.cov_acc = init_imu.cov_acc * (init_imu.N - 1.0) / init_imu.N
                      + (imu.acc_gyro.head<3>() - init_imu.mean_acc).cwiseProduct(imu.acc_gyro.head<3>() - init_imu.mean_acc)
                      * (init_imu.N - 1.0) / (init_imu.N * init_imu.N);
    init_imu.cov_gyr = init_imu.cov_gyr * (init_imu.N - 1.0) / init_imu.N
                      + (imu.acc_gyro.tail<3>() - init_imu.mean_gyr).cwiseProduct(imu.acc_gyro.tail<3>() - init_imu.mean_gyr)
                      * (init_imu.N - 1.0) / (init_imu.N * init_imu.N);
    init_imu.N++;
  }

  init_imu.done = (init_imu.N > N_INIT);
  return init_imu.done;
}

void imu_init_finalize(
    const ImuInit&         imu_init,
    const Eigen::Vector3d& Lidar_T_wrt_IMU,
    const Eigen::Matrix3d& Lidar_R_wrt_IMU,
    NavStateLidar&         ukf_state){

    // Gravità: acc_normalizzata * 9.81
    ukf_state.gravity() = - imu_init.mean_acc / imu_init.mean_acc.norm() * 9.81;

    // Bias giroscopio: media delle misure a robot fermo
    ukf_state.biasGyro() = imu_init.mean_gyr;

    // Estrinseci LiDAR-IMU
    ukf_state.R_li() = Lidar_R_wrt_IMU;
    ukf_state.t_li() = Lidar_T_wrt_IMU;

    Matrix24d P = Matrix24d::Zero();
    P.diagonal().segment<9>(0).setConstant(1.0);// rot, pos, vel (0..8)
    P.diagonal().segment<3>(9).setConstant(1e-4);// bg (9..11)
    P.diagonal().segment<3>(12).setConstant(1e-3);// ba (12..14)
    P.diagonal().segment<3>(15).setConstant(1e-5);// g (15..17)
    P.diagonal().segment<3>(18).setConstant(1e-5);// R_li (18..20)
    P.diagonal().segment<3>(21).setConstant(1e-5);// t_li (21..23)

    // Stampa di diagnostica (sostituisce ROS_INFO)
    std::cout << "[imu_init] Done after " << imu_init.N << " imu samples.\n"
              << "  grav  : " << ukf_state.gravity().transpose()      << "\n"
              << "  bg    : " << ukf_state.biasGyro().transpose()     << "\n"
              << " |acc|  : " << imu_init.mean_acc.norm()             << "\n";
}

bool first_cloud_ever = true, cloud_check = false;

void ukf_pred_and_undistort(
    UKFImuLidar&                       ukf,
    NavStateLidar&                     ukf_state,
    ImuInputs&                         imu_measures,
    LidarMeasurement&                  lidar_meas,
    ImuMeasurement&                    last_imu_meas,
    double&                            last_lidar_time,
    Eigen::Matrix<double, 12, 12>&     Q,
    ContainerType&                     pcl_out, 
    Eigen::Vector3d                    t_LI, 
    Eigen::Matrix3d                    R_LI){

  // Ordina i punti LiDAR per timestamp relativo
  PointCloud cloud = *(lidar_meas.cloud);
  if(first_cloud_ever){
    std::cout << "\n--- BEFORE SORT ---\n";
    for (size_t i = 1; i < cloud.points.size(); ++i){
      if(cloud.points[i].curvature<cloud.points[i-1].curvature){
        std::cout << " founded non ordered points at position i="<< i <<"\n";
        std::cout << " curvature[i-1]=" << cloud.points[i-1].curvature << "\n";
        std::cout << " curvature[i]=" << cloud.points[i].curvature << "\n";
        cloud_check = true;
        break; // interested only in knowing if there is almost 1 element out of order
      }
    }
    std::cout << " [time-check]  Timestamp delle misure imu: \n";
    for (auto imu : imu_measures){
      std::cout << "    " << imu.timestamp << "\n" ;
    }
    std::cout << " [time-check]  Timestamp delle misure lidar: \n";
    std::cout << " lidar.timestamp: " << lidar_meas.timestamp << "\n";
    std::cout << " lidar.end_time: " << lidar_meas.end_time << "\n";
  }
  
  std::sort(cloud.points.begin(), cloud.points.end(), [](const PointType& a, const PointType& b) { return a.curvature < b.curvature; });
                                      
  if(cloud_check && first_cloud_ever){
    std::cout << "\n--- AFTER SORT ---\n";
    for (size_t i = 1; i < cloud.points.size(); ++i){
      if(cloud.points[i].curvature<cloud.points[i-1].curvature){
        std::cout << "--- founded non ordered points at position i="<< i <<"\n";
        std::cout << "--- curvature[i-1]=" << cloud.points[i-1].curvature << "\n";
        std::cout << "--- curvature[i]=" << cloud.points[i].curvature << "\n";
        break;
      }
    }
    std::cout << " all points are successfully sorted  "<<"\n";
    first_cloud_ever = false;
  }

  pcl_out.clear();
  pcl_out.reserve(cloud.size());

  // pose IMU tra scan lidar

  std::vector<ImuPose> imu_poses;
  ImuPose imu_pos;
  imu_poses.reserve(imu_measures.size() + 1); // +1 initial pose
  imu_pos.rot = ukf_state.rotation();
  imu_pos.pos = ukf_state.position();
  imu_pos.vel = ukf_state.velocity();
  imu_pos.acc = last_imu_meas.acc_gyro.head(3);
  imu_pos.gyr = last_imu_meas.acc_gyro.tail(3);
  imu_pos.offset_time = 0.0;
  imu_poses.push_back(imu_pos);

  // Forward propagation with trapezoidal integration between IMU inputs
  Vector6d in_avr;
  double dt = 0.0;
  size_t i = 0;

  for ( i=0 ; i < imu_measures.size(); ++i){
    // Media trapezoidale tra due IMU consecutivi
    if (imu_measures[i].timestamp < last_lidar_time){
      std::cout<<" [forward-prop] è stato rilevato un timestamp di tail minore di last_lidar_time \n";
      continue;  // non capita mai...
    }

    if(i == 0){
      in_avr              = 0.5 * (last_imu_meas.acc_gyro + imu_measures[i].acc_gyro);
      dt                  = imu_measures[i].timestamp - last_imu_meas.timestamp;
    }
    /*else if(i == imu_measures.size()-1){
      in_avr              = 0.5 * (imu_measures[i-1].acc_gyro + imu_measures[i].acc_gyro);
      dt                  = lidar_meas.end_time - imu_measures[i-1].timestamp;
    }*/
    else {
      in_avr              = 0.5 * (imu_measures[i-1].acc_gyro + imu_measures[i].acc_gyro);
      dt                  = imu_measures[i].timestamp - imu_measures[i-1].timestamp;
    }

    if(dt <= 0 ){ 
      std::cout << " [forward-prop] rilevato dt <= 0"; // non capita mai...
    }

    // Normalizza l'accelerazione rispetto alla gravità stimata
    //in_avr.head(3) = in_avr.head(3) * 9.81 / in_avr.head(3).norm();

    ukf.propagate(ukf_state, in_avr, Q, dt);

    // Salva posa intermedia per la backward propagation
    imu_pos.offset_time = imu_measures[i].timestamp - lidar_meas.timestamp;
    imu_pos.rot         = ukf_state.rotation();
    imu_pos.pos         = ukf_state.position();
    imu_pos.vel         = ukf_state.velocity();
    imu_pos.acc         = ukf_state.rotation() * (in_avr.head(3) - ukf_state.biasAcc()) + ukf_state.gravity();
    imu_pos.gyr         = in_avr.tail(3) - ukf_state.biasGyro();
    imu_poses.push_back(imu_pos);
  }

  /*if (lidar_meas.end_time < imu_measures.back().timestamp) {
    std::cout << "/ è stato rilevato un pacchetto con imu_end_time >= lidar_end_time /";
  }// okay, quindi può capitare*/

  // Backward propagation - COMPENSATION
  if (cloud.points.empty()) return;

  Eigen::Matrix3d R_end = ukf_state.rotation();
  Eigen::Vector3d p_end = ukf_state.position();

  auto it_pcl = cloud.points.end() - 1;

  for (auto it_imu = imu_poses.end() - 1; it_imu != imu_poses.begin(); --it_imu) {
    const ImuPose& head = *(it_imu - 1);
    const ImuPose& tail = *it_imu;

    const Eigen::Matrix3d& R_imu    = head.rot;
    const Eigen::Vector3d& vel_imu  = head.vel;
    const Eigen::Vector3d& pos_imu  = head.pos;
    const Eigen::Vector3d& acc_imu  = tail.acc;
    const Eigen::Vector3d& gyr_imu  = tail.gyr;

    // Per ogni punto nel segmento temporale [head, tail] scartando i punti al di fuori dell'ultima IMU-pose perchè ricostruiti male
    for (; it_pcl->curvature / 1000.0 > head.offset_time; --it_pcl) {

      if (it_pcl->curvature / 1000.0 > tail.offset_time) continue;

      dt = it_pcl->curvature / 1000.0 - head.offset_time;

      // Rotazione al timestamp del punto
      Eigen::Matrix3d R_i = R_imu * lie_algebra::SO3Exp(gyr_imu * dt);

      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);

      // Traslazione relativa tra la posa al tempo i
      // e la posa alla fine della scansione
      Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - p_end;

      // Formula completa di compensazione (include estrinseci)
      // P_compensate = R_LI^T * ( R_end^T * ( R_i * (R_LI*P_i + T_LI) + T_ei ) - T_LI )
      Eigen::Vector3d P_compensate =
          //ukf_state.R_li().transpose() * (R_end.transpose() * (R_i * (ukf_state.R_li() * P_i + ukf_state.t_li()) + T_ei) - ukf_state.t_li());
          R_LI.transpose() * (R_end.transpose() * (R_i * (R_LI * P_i + t_LI) + T_ei) - t_LI); // R_LI.transpose() * 
      
      //std::cout<<" --- punto compensato da: "<< P_i.transpose()<<" a "<<P_compensate.transpose()<<"\n";

      //pcl_out->x = static_cast<float>(P_compensate(0));
      //pcl_out->y = static_cast<float>(P_compensate(1));
      //pcl_out->z = static_cast<float>(P_compensate(2));
      pcl_out.emplace_back(static_cast<float>(P_compensate(0)), static_cast<float>(P_compensate(1)), static_cast<float>(P_compensate(2)));

      if (it_pcl == cloud.points.begin()) break;
    }
    if (it_pcl == cloud.points.begin()) break;
  }
}


void init_pipeline(Pipeline& pipeline, const LidarMeasurement& lidar_meas){
  ContainerType pcl;
  pcl.reserve(lidar_meas.cloud->size());

  for (const auto& pt : lidar_meas.cloud->points){
    pcl.emplace_back(pt.x, pt.y, pt.z);
  }

  pipeline.compute(lidar_meas.timestamp, pcl);
}

int main(int argc, char** argv){
  if (argc < 3) {
    std::cout << "usage: this-executable path-to-output trajectory-type" << std::endl;
    std::cout << "example: ./LIO-UKF-MADICP /home/giordano/Desktop/UKF_ICP 1" << std::endl;
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
      dataset_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/quad_easy/ros2";
      break;
    case 2:
      dataset_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/quad_medium/ros2";
      break;
    case 3:
      dataset_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/quad_hard/ros2";
      break;
    case 4:
      dataset_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/stairs/ros2";
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
  
  int INIT_STEPS  = 10;   // numero di misure IMU per l'inizializzazione

  ImuInputs imu_inputs;
  LidarInputs lidar_inputs;
  std::vector<int> imu_indices;

  //std::string gt_filename = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/ground-truth/tum_format/gt-nc-quad-easy.csv";

  read_ros2_bag_imu(imu_inputs, lidar_inputs, dataset_filename);
  //read_states(dts_states, gt_states, gt_filename);

  // SYNCHRONIZATION MEASURES LIDAR IMU
  double t_start = std::max(imu_inputs.front().timestamp, lidar_inputs.front().timestamp);
  double t_end   = std::min(imu_inputs.back().timestamp ,  lidar_inputs.back().timestamp);
  std::cout << "[SYNC] intervallo comune: " << t_start << " -> " << t_end << "\n";

  // PRIMO INDICE LIDAR VALIDO
  int lidar_start_idx = 0;
  while (lidar_start_idx < (int)lidar_inputs.size() && lidar_inputs[lidar_start_idx].timestamp < t_start) {
    ++lidar_start_idx;
  }
  std::cout << "[SYNC] primo indice LIDAR: " << lidar_start_idx << "\n";

  // TROVA PRIMO INDICE IMU CORRISPONDENTE
  int imu_start_idx = 0;
  while (imu_start_idx < (int)imu_inputs.size() && imu_inputs[imu_start_idx].timestamp < lidar_inputs[lidar_start_idx].timestamp) { 
    ++imu_start_idx; 
  }
  std::cout << "[SYNC] primo indice IMU: " << imu_start_idx << "\n";

  // trasformazione T_li ricavata dal dataset:  LIDAR -> IMU  (esprime le coordinate del frame LIDAR rispetto l'IMU frame)
  // p_imu  =  T_li * p_lidar  =  R_li * p_lidar + t_li
  /*R_LI << 1,0,0,
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
  std::cout << "imu_inputs.size()     = " << imu_inputs.size()     << "\n";
  std::cout << "lidar_inputs.size() = " << lidar_inputs.size() << "\n";
  //std::cout << "gt_states.size()      = " << gt_states.size()      << "\n";
  //std::cout << "dts_states.size()     = " << dts_states.size()     << "\n";

  NavStateLidar ukf_state;

  Matrix12d Q = Matrix12d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);  // sigma_noise_accel
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);  // sigma_noise_gyro
  Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);  // sigma_walk_accel
  Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);  // sigma_walk_gyro
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  ukf_state.rotation() = Eigen::Matrix3d::Identity();
  ukf_state.velocity() = Eigen::Vector3d(0, 0, 0);
  ukf_state.position() = Eigen::Vector3d(0, 0, 0);
  ukf_state.biasAcc()  = Eigen::Vector3d(0, 0, 0);
  ukf_state.biasGyro() = Eigen::Vector3d(0, 0, 0);


  // ------- UKF filter
  UKFImuLidar ukf_lidar;
  ukf_lidar.setWeights(ukf_state.stateDim(), 12, alpha);

  // -------- initializing estimates
  int N = lidar_inputs.size() - lidar_start_idx;  

  std::vector<NavStateLidar> estimates(N-1);
  std::vector<double> dts(N+1);

  OdomObs obs;
  Eigen::MatrixXd R, H;
  Eigen::Isometry3d Z;
  Vector6d z_tang;
  //double prev_lidar_time=0.0;

  std::cout << "=== STARTING FILTER LOOP ===\n";

  bool imu_initialized = false;
  int imu_idx = 0, lidar_idx, l=0; // n_pose_stamped = 0, imu_idx = imu_start_idx
  double last_lidar_time = 0.0;
  ImuInit init_imu;
  ImuMeasurement last_imu_meas;
  ContainerType undistort_cloud;
  double Scale_Factor = 0.0; // scaling factor for weighting filter updates

  for (lidar_idx = lidar_start_idx; lidar_idx < (int)lidar_inputs.size()-1; ++lidar_idx){

    if (lidar_idx%100 == 0){
      if (l%2 == 0){
        std::cout << "\n--- iteration: " << lidar_idx << " ---";
      }
      else{
        std::cout << "   iteration: " << lidar_idx << " ---";
      }
      l++;
    } 

    LidarMeasurement lidar_meas = lidar_inputs[lidar_idx];
    if (lidar_idx == lidar_start_idx){
        // first lidar frame, skip measure
        estimates[0] = ukf_state;
        dts[0] = lidar_meas.timestamp;
        last_lidar_time = lidar_meas.end_time;
        continue;
    }

    // cumulate IMU measures between lidar frames
    ImuInputs imu_pack;
    while (imu_inputs[imu_idx].timestamp <= lidar_inputs[lidar_idx+1].timestamp){
        imu_pack.push_back(imu_inputs[imu_idx]);
        ++imu_idx;
    }

    if (!imu_initialized) {
      bool done = imu_init_accumulate(imu_pack, init_imu, INIT_STEPS);
      if (done) {

        imu_init_finalize(init_imu, t_LI, R_LI, ukf_state);
        init_pipeline(pipeline, lidar_meas);  // initialize internal map with initial state as Identity: used with undistortion 

        estimates[lidar_idx-lidar_start_idx] = ukf_state;
        dts[lidar_idx-lidar_start_idx] = lidar_meas.timestamp;

        last_imu_meas = imu_pack.back();
        last_lidar_time = lidar_meas.end_time;
        imu_initialized = true;
      }
      continue;
    }

    ukf_pred_and_undistort(ukf_lidar, ukf_state, imu_pack, lidar_meas, last_imu_meas, last_lidar_time, Q, undistort_cloud, t_LI, R_LI);

    // passo la cloudpoint a MAD-ICP
    pipeline.compute(lidar_meas.timestamp, undistort_cloud);

    // ottengo la trasformazione rispetto alla posizione precedente e la converto nello spazio tangente
    Z = pipeline.currentPose();
    z_tang.head(3) = Z.translation();
    z_tang.tail(3) = lie_algebra::SO3Log(Z.rotation() * R_LI);
    // ricavo le informazioni da mad-icp per calcolare la matrice di covarianza
    H = pipeline.getHinfo();
    H = H + 1e-6 * Matrix6d::Identity();   // termine correttivo per evitare errori numerici
    R = H.inverse() * Scale_Factor;

    if (!R.allFinite()) {
      std::cout << "NaN estimate at lidar index =" << lidar_idx << "\n";
      break;
    }

    /*ukf_lidar.write_log("162513");
    ukf_lidar.write_log(lidar_meas.timestamp);
    ukf_lidar.write_pipe("162513");
    ukf_lidar.write_pipe(lidar_meas.timestamp);
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
    ukf_lidar.write_pipe("\n");*/

    obs.meas() = Z;
    ukf_lidar.update(ukf_state, obs, R);

    // save states
    estimates[lidar_idx-lidar_start_idx] = ukf_state;
    dts[lidar_idx-lidar_start_idx] = lidar_meas.timestamp;

    if (!ukf_state.position().allFinite()) {
      std::cout << "NaN estimate for ukf_lidar at lidar_idx=" << lidar_idx << "\n";
      break;
    }

    // update 
    last_imu_meas = imu_pack.back();
    last_lidar_time = lidar_meas.end_time;
  }

  // output file
  auto write_traj = [&](const std::string& fname, const std::vector<NavStateLidar>& states, const std::vector<double>& dts, const Vector15d& vec, int prec){
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

  std::cout << "\n\n";
  
  std::string out = std::string(argv[1]);
  Vector15d vect;
  //write_traj(out+"/gt_states.txt", gt_states, dts);

  vect << 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  write_traj(out+"/UKF_LIO_s.txt",  estimates, dts, vect, 6);

  //vect << 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  //write_traj_ext(out+"/est_lio_ext.txt",  estimates_extended, dts, vect, 6);

  vect << 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0; 
  //write_traj(out+"/covariance_r_p_v_und.txt",  estimates, dts, vect, 9);

  vect << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0; 
  //write_traj(out+"/covariance_bg_ba_g_und.txt",  estimates, dts, vect, 9);

  vect << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1; 
  //write_traj(out+"/covariance_rli_tli_und.txt",  estimates, dts, vect, 9);

  vect << 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0; 
  //write_traj(out+"/bg_ba_grav_und.txt",  estimates, dts, vect, 9);

  // ENDING TIME
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Tempo esecuzione: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  std::cout << "job finished." << std::endl;

  return 0;
}
