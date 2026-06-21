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
#include <chrono>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/transform_broadcaster.h>

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
};
using LidarInputs = std::vector<LidarMeasurement>;

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

      // Traslazione relativa tra la posa al tempo i e la posa alla fine della scansione
      Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - p_end;

      // Formula completa di compensazione (include estrinseci)
      // P_compensate = R_LI^T * ( R_end^T * ( R_i * (R_LI*P_i + T_LI) + T_ei ) - T_LI )
      Eigen::Vector3d P_compensate =
          R_LI.transpose() * (R_end.transpose() * (R_i * (R_LI * P_i + t_LI) + T_ei) - t_LI); // R_LI.transpose() * 
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

class UkfNode : public rclcpp::Node {
public:
  UkfNode() : Node("ukf_lio_node"),pipeline_(10.0, false, 0.2, 0.1, 0.8, 0.1, 0.02, 4, 4, false) {

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 1000);

    rclcpp::QoS qos(1000);
    qos.keep_last(1000);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud", qos);

    //cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud", 1000);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    loadDatasetAndInit();

    worker_thread_ = std::thread(&UkfNode::runPipeline, this);
  }

  ~UkfNode() {
    running_ = false;
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  void runPipeline();
  void publishOdom(const NavStateLidar& state, double t);
  void publishCloud(const ContainerType& cloud, double t);
  void publishTF(const NavStateLidar& s, double t);
  void loadDatasetAndInit();

private:
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::thread worker_thread_;
  std::atomic<bool> running_{true};

  ImuInputs   imu_inputs_;
  LidarInputs lidar_inputs_;
  std::string dataset_filename_ = "/home/giordano/Documents/NewerCollegeDataset/Collection_1/quad_hard/ros2";

  Eigen::Matrix3d R_LI_;
  Eigen::Vector3d t_LI_;
  Matrix12d       Q_;
  Vector4d        imu_noise_std_;

  UKFImuLidar   ukf_lidar_;
  NavStateLidar ukf_state_;

  Pipeline pipeline_;

  OdomObs           obs_;
  Eigen::MatrixXd   R_, H_;
  Eigen::Isometry3d Z_;
  Vector6d          z_tang_;

  bool           imu_initialized_ = false;
  int            imu_idx_ = 0, lidar_idx_, l_=0, lidar_start_idx_, init_steps_ = 10;
  double         last_lidar_time_ = 0.0;
  ImuInit        init_imu_;
  ImuMeasurement last_imu_meas_;
  ContainerType  undistort_cloud_;
  double         Scale_Factor_ = 0.0; // scaling factor for weighting filter updates
};

void UkfNode::publishOdom(const NavStateLidar& s, double t){
  nav_msgs::msg::Odometry msg;

  msg.header.stamp = rclcpp::Time(t * 1e9); // se seconds
  msg.header.frame_id = "map";
  msg.child_frame_id = "lidar";

  msg.pose.pose.position.x = s.position().x();
  msg.pose.pose.position.y = s.position().y();
  msg.pose.pose.position.z = s.position().z();

  Eigen::Quaterniond q(s.rotation());
  q.normalize();

  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.pose.orientation.w = q.w();

  odom_pub_->publish(msg);
}

sensor_msgs::msg::PointCloud2 toROSMsg(const pcl::PointCloud<PointType>& cloud, const std::string& frame, double t){
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);

  msg.header.frame_id = frame;
  msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));

  return msg;
}

void UkfNode::publishCloud(const ContainerType& cloud, double t){
  if (cloud.empty()) return;
  pcl::PointCloud<PointType> pcl_cloud;
  pcl_cloud.reserve(cloud.size());

  for (const auto& p : cloud) {
    PointType pt;
    pt.x = static_cast<float>(p.x());
    pt.y = static_cast<float>(p.y());
    pt.z = static_cast<float>(p.z());
    pt.intensity = 0.0f;
    pt.normal_x = 0.0f;
    pt.normal_y = 0.0f;
    pt.normal_z = 0.0f;
    pt.curvature = 0.0f;

    pcl_cloud.push_back(pt);
  }

  pcl_cloud.width = pcl_cloud.size();
  pcl_cloud.height = 1;
  pcl_cloud.is_dense = true;

  cloud_pub_->publish(toROSMsg(pcl_cloud, "lidar", t));
}

void UkfNode::publishTF(const NavStateLidar& s, double t){
  geometry_msgs::msg::TransformStamped tf_msg;

  tf_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
  tf_msg.header.frame_id = "map";
  tf_msg.child_frame_id = "lidar"; // base_link

  tf_msg.transform.translation.x = s.position().x();
  tf_msg.transform.translation.y = s.position().y();
  tf_msg.transform.translation.z = s.position().z();

  Eigen::Quaterniond q(s.rotation());
  q.normalize();

  tf_msg.transform.rotation.x = q.x();
  tf_msg.transform.rotation.y = q.y();
  tf_msg.transform.rotation.z = q.z();
  tf_msg.transform.rotation.w = q.w();

  tf_broadcaster_->sendTransform(tf_msg);
}

void UkfNode::loadDatasetAndInit(){

  read_ros2_bag_imu(imu_inputs_, lidar_inputs_, dataset_filename_);

  // SYNCHRONIZATION MEASURES LIDAR IMU
  double t_start = std::max(imu_inputs_.front().timestamp, lidar_inputs_.front().timestamp);
  double t_end   = std::min(imu_inputs_.back().timestamp ,  lidar_inputs_.back().timestamp);
  std::cout << "[SYNC] intervallo comune: " << t_start << " -> " << t_end << "\n";

  // first valid lidar index
  lidar_start_idx_ = 0;
  while (lidar_start_idx_ < (int)lidar_inputs_.size() && lidar_inputs_[lidar_start_idx_].timestamp < t_start) {
    ++lidar_start_idx_;
  }
  std::cout << "[SYNC] primo indice LIDAR valido: " << lidar_start_idx_ << "\n";

  std::cout << "=== POST READING FUNCTIONS ===\n";
  std::cout << "imu_inputs_.size()     = " << imu_inputs_.size() << "\n";
  std::cout << "lidar_inputs_.size() = " << lidar_inputs_.size() << "\n";

  // trasformazione T_li ricavata dal dataset:  LIDAR -> IMU  (esprime le coordinate del frame LIDAR rispetto l'IMU frame)
  // p_imu  =  T_li * p_lidar  =  R_li * p_lidar + t_li

  // for alphasense IMU and oster pcl
  /*R_LI << 1,0,0,
          0,-1,0,
          0,0,-1;
  t_LI << -0.065, -0.00855, -0.0336;*/
  
  // for oster IMU and oster pcl
  R_LI_ << 1,0,0,
          0,1,0,
          0,0,1;
  t_LI_ << 0.0, 0.0, 0.0;

  imu_noise_std_<< 1.86e-03,   // sigma_noise_accel - 8e-2
                   1.87e-04,   // sigma_noise_gyro  - 4e-3
                   4.33e-04,   // sigma_walk_accel  - 4e-5
                   2.66e-05;   // sigma_walk_gyro   - 2e-6

  Q_ = Matrix12d::Identity();
  Q_.block<3, 3>(0, 0) *= pow(imu_noise_std_(0), 2);  // sigma_noise_accel
  Q_.block<3, 3>(3, 3) *= pow(imu_noise_std_(1), 2);  // sigma_noise_gyro
  Q_.block<3, 3>(6, 6) *= pow(imu_noise_std_(2), 2);  // sigma_walk_accel
  Q_.block<3, 3>(9, 9) *= pow(imu_noise_std_(3), 2);  // sigma_walk_gyro
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  ukf_state_.rotation() = Eigen::Matrix3d::Identity();
  ukf_state_.velocity() = Eigen::Vector3d(0, 0, 0);
  ukf_state_.position() = Eigen::Vector3d(0, 0, 0);
  ukf_state_.biasAcc()  = Eigen::Vector3d(0, 0, 0);
  ukf_state_.biasGyro() = Eigen::Vector3d(0, 0, 0);

  // ------- UKF filter
  ukf_lidar_.setWeights(ukf_state_.stateDim(), 12, alpha);


  // questi verranno sostituiti con scritture dirette su file durante pubblicazione... 
  // -------- initializing estimates
  //int N = lidar_inputs_.size() - lidar_start_idx;  
  //std::vector<NavStateLidar> estimates(N-1);
  //std::vector<double> dts(N-1);
}

void UkfNode::runPipeline(){

  RCLCPP_INFO(get_logger(), "UKF pipeline started");
  std::cout << "=== STARTING FILTER LOOP ===\n";

  for (lidar_idx_ = lidar_start_idx_; lidar_idx_ < (int)lidar_inputs_.size()-1; ++lidar_idx_){

    auto t8 = std::chrono::high_resolution_clock::now();

    if (lidar_idx_ % 100 == 0){
      if (l_ % 1 == 0){
        std::cout << "\n --- iteration: " << lidar_idx_ << " ---";
      }
      else{
        std::cout << "   iteration: " << lidar_idx_ << " ---";
      }
      l_++;
    } 

    LidarMeasurement lidar_meas = lidar_inputs_[lidar_idx_];
    if (lidar_idx_ == lidar_start_idx_){
      // first lidar frame, skip measure

      // anche qui bisogna adattare da scrittura su struttura dati a scrittura diretta su file
      //estimates[0] = ukf_state;
      //dts[0] = lidar_meas.timestamp;
      last_lidar_time_ = lidar_meas.end_time;
      continue;
    }

    // cumulate IMU measures between lidar frames
    ImuInputs imu_pack;
    while (imu_inputs_[imu_idx_].timestamp <= lidar_inputs_[lidar_idx_+1].timestamp){
      imu_pack.push_back(imu_inputs_[imu_idx_]);
      ++imu_idx_;
    }

    if (!imu_initialized_) {
      bool done = imu_init_accumulate(imu_pack, init_imu_, init_steps_);
      if (done) {

        imu_init_finalize(init_imu_, t_LI_, R_LI_, ukf_state_);
        init_pipeline(pipeline_, lidar_meas);  // initialize internal map with initial state as Identity: used with undistortion 

        // anche qui bisogna adattare da scrittura su struttura dati a scrittura diretta su file
        //estimates[lidar_idx_-lidar_start_idx_] = ukf_state;
        //dts[lidar_idx_-lidar_start_idx_] = lidar_meas.timestamp;

        last_imu_meas_   = imu_pack.back();
        last_lidar_time_ = lidar_meas.end_time;
        imu_initialized_ = true;

        publishOdom(ukf_state_, lidar_meas.timestamp);
        publishTF(ukf_state_, lidar_meas.timestamp);
        //publishCloud(lidar_meas, lidar_meas.timestamp);
      }
      continue;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    ukf_pred_and_undistort(ukf_lidar_, ukf_state_, imu_pack, lidar_meas, last_imu_meas_, last_lidar_time_, Q_, undistort_cloud_, t_LI_, R_LI_);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "ukf_pred_and_undistort(): " << elapsed_ms << " ms" << std::endl;

    // passo la cloudpoint a MAD-ICP
    auto t2 = std::chrono::high_resolution_clock::now();
    pipeline_.compute(lidar_meas.timestamp, undistort_cloud_);
    auto t3 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Pipeline.compute(): " << elapsed_ms << " ms" << std::endl;

    // ottengo la trasformazione rispetto alla posizione precedente e la converto nello spazio tangente
    Z_ = pipeline_.currentPose();
    z_tang_.head(3) = Z_.translation();
    z_tang_.tail(3) = lie_algebra::SO3Log(Z_.rotation() * R_LI_);
    // ricavo le informazioni da mad-icp per calcolare la matrice di covarianza
    H_ = pipeline_.getHinfo();
    H_ = H_ + 1e-6 * Matrix6d::Identity();   // termine correttivo per evitare errori numerici
    R_ = H_.inverse() * Scale_Factor_;

    if (!R_.allFinite()) {
      std::cout << "NaN estimate at lidar index =" << lidar_idx_ << "\n";
      break;
    }

    /*
    ukf_lidar_.write_log("162513");
    ukf_lidar_.write_log(lidar_meas.timestamp);
    ukf_lidar_.write_pipe("162513");
    ukf_lidar_.write_pipe(lidar_meas.timestamp);
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(z_tang_(0));
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(z_tang_(1));
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(z_tang_(2));
    Eigen::Quaterniond q(Z_.rotation());
    q.normalize();
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(q.x());
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(q.y());
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(q.z());
    ukf_lidar_.write_pipe(" ");
    ukf_lidar_.write_pipe(q.w());
    ukf_lidar_.write_pipe("\n");*/

    obs_.meas() = Z_;

    auto t4 = std::chrono::high_resolution_clock::now();
    ukf_lidar_.update(ukf_state_, obs_, R_);
    auto t5 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    std::cout << "ukf_lidar_.update(): " << elapsed_ms << " ms" << std::endl;

    // save states
    // anche qui bisogna adattare da scrittura su struttura dati a scrittura diretta su file
    //estimates[lidar_idx-lidar_start_idx] = ukf_state;
    //dts[lidar_idx-lidar_start_idx] = lidar_meas.timestamp;

    if (!ukf_state_.position().allFinite()) {
      std::cout << "NaN estimate for ukf_lidar at lidar_idx=" << lidar_idx_ << "\n";
      break;
    }

    auto t6 = std::chrono::high_resolution_clock::now();
    publishOdom(ukf_state_, lidar_meas.timestamp);
    publishTF(ukf_state_, lidar_meas.timestamp);
    publishCloud(undistort_cloud_, lidar_meas.timestamp);
    auto t7 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
    std::cout << "Publishing time (): " << elapsed_ms << " ms" << std::endl;

    // update 
    last_imu_meas_ = imu_pack.back();
    last_lidar_time_ = lidar_meas.end_time;

    auto t9 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
    std::cout << "loop time (): " << elapsed_ms << " ms" << "\n\n";
  }
}

int main(int argc, char **argv){

  rclcpp::init(argc, argv);

  auto node = std::make_shared<UkfNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}