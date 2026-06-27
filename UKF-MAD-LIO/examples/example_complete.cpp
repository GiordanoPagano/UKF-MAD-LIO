#include <ukf_manifold/lie_algebra.h>
#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>

#include <fstream>
#include <istream>

#include <chrono>
#include <thread>

#include <string>

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// ------------------------------------ Utilities for generating trajectories: ---------------------------------
//
// ./example_complete /home/giordano/Desktop/traj 1 10 1-1-1-1
//
// ./this-executable  path-to-output  trajectory-type  trajectory-amplitude  bias-multiplier
//
// ------------------------------------------- Utilities for plotting: ----------------------------------------
//
// splot "traj_gt.txt" using 1:2:3 w l, "traj_est_ext_dirt.txt" using 1:2:3 w l, "traj_est_ext_bias.txt" using 1:2:3 w l, 
//       "traj_est_dirt.txt" using 1:2:3 w l, "traj_est_bias.txt" using 1:2:3 w l, "traj_est.txt" using 1:2:3 w l
//
// splot "biases_est_dirt.txt" using 1:2:3 w l, "biases_est_bias.txt" using 1:2:3 w l, "biases_gt.txt" using 1:2:3 w l
//
// splot "imu_inputs_circular_dirt.txt" using 2:3:4 w l, "imu_inputs_circular_bias.txt" using 2:3:4 w l, "imu_inputs_circular.txt" using 2:3:4 w l
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string data_folder(UKF_DATA_FOLDER);

using namespace ukf_manifold;

using ImuStates        = std::vector<NavState>;
using ImuStatesExt     = std::vector<NavStateExtended>;
using ImuInputs        = std::vector<Vector6d>;
using GpsMeasurements  = std::map<int, Eigen::Vector3d>;
using IntegrationSteps = std::vector<double>; 

using Vector4d    = Eigen::Matrix<double, 4, 1>;
using Matrix12d   = Eigen::Matrix<double, 12, 12>;

// ====== std::cout << "DEBUG" << std::endl; ======

void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, ImuInputs& inputs_bias_, ImuInputs& inputs_dirt_, 
                       const Vector4d& imu_noise_std_, const std::string& traj_name_, const std::string bias_gt_str,
                       const int T, const double radius, const int imu_freq, const int N, const double dt);

void simulate_gps_data(GpsMeasurements& zs_gps, const ImuStates& states, const double& gps_noise_std, const int gps_freq);

void read_input(IntegrationSteps& dts_, ImuInputs& inputs_, ImuInputs& inputs_bias_, ImuInputs& inputs_dirt_, const std::string filename, 
                const std::string filename_bias, const std::string filename_dirt) {
 
   // ====== INPUT: [acc, gyro]
  // ------ [inputs] = GT + NOISE
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double dt, ax, ay, az, wx, wy, wz;
    is >> dt >> ax >> ay >> az >> wx >> wy >> wz;
    Vector6d input;
    input << ax, ay, az, wx, wy, wz;
    dts_.push_back(dt);
    inputs_.push_back(input);
  }
  inputs_.shrink_to_fit();
  dts_.shrink_to_fit();
  // ------ [inputs] = GT + BIAS
  std::ifstream iss(filename_bias.c_str());
  if (!iss.good())
    return;
  while (iss) {
    double dt, ax, ay, az, wx, wy, wz;
    iss >> dt >> ax >> ay >> az >> wx >> wy >> wz;
    Vector6d input;
    input << ax, ay, az, wx, wy, wz;
    inputs_bias_.push_back(input);
  }
  inputs_bias_.shrink_to_fit();
  // ------ [inputs] = GT + NOISE + BIAS
  std::ifstream isss(filename_dirt.c_str());
  if (!isss.good())
    return;
  while (isss) {
    double dt, ax, ay, az, wx, wy, wz;
    isss >> dt >> ax >> ay >> az >> wx >> wy >> wz;
    Vector6d input;
    input << ax, ay, az, wx, wy, wz;
    inputs_dirt_.push_back(input);
  }
  inputs_dirt_.shrink_to_fit();
}

void read_states(ImuStates& states, const std::string filename, ImuStatesExt& states_ext) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double dt, rx, ry, rz, vx, vy, vz, px, py, pz;
    is >> dt >> rx >> ry >> rz >> vx >> vy >> vz >> px >> py >> pz;
    const Eigen::Vector3d rot_vec = Eigen::Vector3d(rx, ry, rz);
    const Eigen::Matrix3d R       = lie_algebra::SO3Exp(rot_vec);
    const Eigen::Vector3d v       = Eigen::Vector3d(vx, vy, vz);
    const Eigen::Vector3d p       = Eigen::Vector3d(px, py, pz);
    const Eigen::Vector3d b_acc   = Eigen::Vector3d(0, 0, 0);
    const Eigen::Vector3d b_gyro  = Eigen::Vector3d(0, 0, 0);

    //NavState state;
    NavStateExtended state_e;

    //state.rotation() = R;
    //state.velocity() = v;
    //state.position() = p;
    state_e.rotation() = R;
    state_e.velocity() = v;
    state_e.position() = p;
    state_e.biasAcc() = b_acc;
    state_e.biasGyro() = b_gyro;

    //states.push_back(state);
    states_ext.push_back(state_e);
  }
}



Eigen::Matrix4d umeyama_manual( const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y){
    int N = X.cols();
    Eigen::Vector3d meanX = X.rowwise().mean();
    Eigen::Vector3d meanY = Y.rowwise().mean();
    Eigen::MatrixXd Xc = X.colwise() - meanX;
    Eigen::MatrixXd Yc = Y.colwise() - meanY;
    Eigen::Matrix3d H = Xc * Yc.transpose() / N;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    Eigen::Vector3d t = meanY - R * meanX;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}
double computeATE(const ImuStates& estimates, const ImuStates& gt_){
    // ATE with no kind of adjustment; for evaluating the reconstruction starting from an initial estimate error
    double error = 0;
    for (size_t i = 0; i < gt_.size(); ++i){
      error += (gt_[i].position() - estimates[i].position()).squaredNorm();
    }
    return std::sqrt(error / gt_.size());
}
double computeATE2(const ImuStates& est, const ImuStates& gt){
  // ATE with traj_estimated translated; for evaluating the ability of reconstruction only
    assert(est.size() == gt.size());
    int N = est.size();
    Eigen::MatrixXd P_gt(3, N);
    Eigen::MatrixXd P_est(3, N);
    for(int i=0;i<N;i++){
        P_gt.col(i)  = gt[i].position();
        P_est.col(i) = est[i].position();
    }
    // alignment 
    Eigen::Matrix4d T;
    T = umeyama_manual(P_est, P_gt);
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    double error = 0;
    for(int i=0;i<N;i++){
        Eigen::Vector3d aligned = R * P_est.col(i) + t;
        error += (P_gt.col(i) - aligned).squaredNorm();
    }
    return std::sqrt(error / N);
}
double computeRPET(const ImuStates& est, const ImuStates& gt, int delta = 1){
    assert(est.size() == gt.size());
    int N = est.size();
    double error = 0;
    int count = 0;
    for(int i=0; i < N - delta; ++i){
        Eigen::Matrix4d Tgt1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Tgt2 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Test1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Test2 = Eigen::Matrix4d::Identity();
        Tgt1.block<3,3>(0,0) = gt[i].rotation();
        Tgt1.block<3,1>(0,3) = gt[i].position();
        Tgt2.block<3,3>(0,0) = gt[i+delta].rotation();
        Tgt2.block<3,1>(0,3) = gt[i+delta].position();
        Test1.block<3,3>(0,0) = est[i].rotation();
        Test1.block<3,1>(0,3) = est[i].position();
        Test2.block<3,3>(0,0) = est[i+delta].rotation();
        Test2.block<3,1>(0,3) = est[i+delta].position();
        Eigen::Matrix4d E = Tgt1.inverse() * Tgt2 * (Test1.inverse() * Test2).inverse();
        Eigen::Vector3d trans = E.block<3,1>(0,3);
        error += trans.squaredNorm();
        count++;
    }
    return std::sqrt(error / count);
}
double computeRPER(const ImuStates& est, const ImuStates& gt, int delta = 1){
    double error = 0;
    int N = est.size();
    int count = 0;
    for(int i=0; i < N - delta; ++i){
        Eigen::Matrix3d Rgt = gt[i].rotation().transpose() * gt[i+delta].rotation();
        Eigen::Matrix3d Rest = est[i].rotation().transpose() * est[i+delta].rotation();
        Eigen::Matrix3d E = Rgt * Rest.transpose();
        Eigen::AngleAxisd aa(E);
        error += aa.angle() * aa.angle();
        count++;
    }
    return std::sqrt(error / count);
}
double computeATEext(const ImuStatesExt& estimates, const ImuStates& gt_){
    assert(estimates.size() == gt_.size());
    double error = 0;
    for (size_t i = 0; i < gt_.size(); ++i){
      error += (gt_[i].position() - estimates[i].position()).squaredNorm();
    }
    return std::sqrt(error / gt_.size());
}
double computeATE2ext(const ImuStatesExt& est, const ImuStates& gt){
    assert(est.size() == gt.size());
    int N = est.size();
    Eigen::MatrixXd P_gt(3, N);
    Eigen::MatrixXd P_est(3, N);
    for(int i=0;i<N;i++){
        P_gt.col(i)  = gt[i].position();
        P_est.col(i) = est[i].position();
    }
    // alignment 
    Eigen::Matrix4d T;
    T = umeyama_manual(P_est, P_gt);
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    double error = 0;
    for(int i=0;i<N;i++){
        Eigen::Vector3d aligned = R * P_est.col(i) + t;
        error += (P_gt.col(i) - aligned).squaredNorm();
    }
    return std::sqrt(error / N);
}
double computeRPETExt(const ImuStatesExt& est, const ImuStates& gt, int delta = 1){
    assert(est.size() == gt.size());
    int N = est.size();
    double error = 0;
    int count = 0;
    for(int i=0; i < N - delta; ++i){
        Eigen::Matrix4d Tgt1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Tgt2 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Test1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Test2 = Eigen::Matrix4d::Identity();
        Tgt1.block<3,3>(0,0) = gt[i].rotation();
        Tgt1.block<3,1>(0,3) = gt[i].position();
        Tgt2.block<3,3>(0,0) = gt[i+delta].rotation();
        Tgt2.block<3,1>(0,3) = gt[i+delta].position();
        Test1.block<3,3>(0,0) = est[i].rotation();
        Test1.block<3,1>(0,3) = est[i].position();
        Test2.block<3,3>(0,0) = est[i+delta].rotation();
        Test2.block<3,1>(0,3) = est[i+delta].position();
        Eigen::Matrix4d E = Tgt1.inverse() * Tgt2 * (Test1.inverse() * Test2).inverse();
        Eigen::Vector3d trans = E.block<3,1>(0,3);
        error += trans.squaredNorm();
        count++;
    }
    return std::sqrt(error / count);
}
double computeRPERExt(const ImuStatesExt& est, const ImuStates& gt, int delta = 1){
    assert(est.size() == gt.size());
    double error = 0;
    int N = est.size();
    int count = 0;
    for(int i=0; i < N - delta; ++i){
        Eigen::Matrix3d Rgt = gt[i].rotation().transpose() * gt[i+delta].rotation();
        Eigen::Matrix3d Rest = est[i].rotation().transpose() * est[i+delta].rotation();
        Eigen::Matrix3d E = Rgt * Rest.transpose();
        Eigen::AngleAxisd aa(E);
        error += aa.angle() * aa.angle();
        count++;
    }
    return std::sqrt(error / count);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    ATE  : misura l'errore di stima globale tra est e gt tenendo in considerazione anche eventuali valori di offset
//
//    ATE2 : trova la trasformazione ottimale con Umeyama tale da sovrapporre la est con la gt per calcolare
//           in modo reale l'errore di stima senza tenere conto di offset vari
//
//    RPE  : misura drift relativo tra i e i+delta; serve per capire l'errore di stima in traslazione e rotazione
//           relativo alla stima della posizione i-esima



int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "usage: this-executable path-to-output trajectory-type trajectory-amplitude imu-bias-multiplier" << std::endl;
    std::cout << "example usage: ./example_complete /home/username/Desktop/traj 1 10 1-1-1-1" << std::endl;
  }
  if (argc < 2) {
    return 1;
  }
  
  int T           = 10;  // seq time (s)
  double radius   = 10;  // amplitude for trajectory (m)
  int imu_freq    = 100; // imu dreq (Hz)
  int gps_freq    = 5;   //  GPS drequency (Hz)
  int N           = T * imu_freq;   // num od timestamps
  double dt       = 1.0 / imu_freq; // integration step (s)
  
  if (argc >= 4) {
    try {
      radius = std::stoi(argv[3]);
    }
    catch (const std::exception&) {
      std::cerr << "trajectory-amplitude must be an integer... "<< std::endl;
      return 1;
    }
  }

  // multiplier for biases
  Vector4d FACTOR;  
  FACTOR << 1, 1, 1, 1;
  // chek for multipliers for bias
  if (argc >= 5) {
    std::string factor_str = argv[4];
    std::stringstream ss(factor_str);
    std::string token;
    int i = 0;
    while (std::getline(ss, token, '-')) {
      if (i >= 4) {
        std::cerr << "Too many bias multipliers! Expected 4.\n";
        return 1;
      }
      try {
        FACTOR(i) = std::stod(token);
      }
      catch (...) {
        std::cerr << "Invalid bias multiplier format. Use example: 1-1-10-20\n";
        return 1;
      }
      i++;
    }
    if (i != 4) {
      std::cerr << "Expected exactly 4 bias multipliers.\n";
      return 1;
    }
  }

  // imu noise standard deviation (isotropic) [ACC, GYRO, BIAS_ACC, BIAS_GYRO]
  Vector4d imu_noise_std;
  imu_noise_std << 1.86e-03, 1.87e-04, 4.33e-04, 2.66e-05;     //0.05, 0.01, 0.0001, 0.000001;
  imu_noise_std(0)     *= FACTOR(0);
  imu_noise_std(1)     *= FACTOR(1);
  imu_noise_std(2)     *= FACTOR(2);
  imu_noise_std(3)     *= FACTOR(3);
  double gps_noise_std  = 0.5; // (m)

  // getting dummy data from imu and gps
  ImuStatesExt states_ext;
  ImuStates states;
  ImuInputs inputs, inputs_bias, inputs_dirt;
  IntegrationSteps dts;

  // defining the trajectory
  int Traj = 1;
  if (argc >= 3) {
    try {
      Traj = std::stoi(argv[2]);
    }
    catch (const std::exception&) {
      std::cerr << "trajectory-type must be an integer (1-4):\n" 
                << "            1 -> circular\n"
                << "            2 -> wave\n"
                << "            3 -> infinite\n"
                << "            4 -> spiral\n" << std::endl;
      return 1;
    }
  }
  std::string traj_name;
  switch (Traj) {
    case 1:
      traj_name = "circular";
      break;
    case 2:
      traj_name = "wave";
      break;
    case 3:
      traj_name = "infinite";
      break;
    case 4:
      traj_name = "spiral";
      break;
    default:
      std::cerr << "Invalid trajectory type.\n"
                << "1 -> circular\n"
                << "2 -> wave\n"
                << "3 -> infinite\n"
                << "4 -> spiral\n";
      return 1;
  }
  
  simulate_imu_data(states, inputs, inputs_bias, inputs_dirt, imu_noise_std, traj_name, std::string(argv[1]), T, radius, imu_freq, N, dt);
  
  // I + N:      imu_inputs_circular        imu_inputs_wave        imu_inputs_infinite        imu_inputs_spiral
  // I + B:      imu_inputs_circular_bias   imu_inputs_wave_bias   imu_inputs_infinite_bias   imu_inputs_spiral_bias
  // I + N + B:  imu_inputs_circular_dirt   imu_inputs_wave_dirt   imu_inputs_infinite_dirt   imu_inputs_spiral_dirt
  // STATE-GT:   imu_states_gt_circular     imu_states_gt_wave     imu_states_gt_infinite     imu_states_gt_spiral

  std::string inputs_data_path      = data_folder + "/imu_inputs_" + traj_name + ".txt";
  std::string inputs_bias_data_path = data_folder + "/imu_inputs_" + traj_name + "_bias.txt";
  std::string inputs_dirt_data_path = data_folder + "/imu_inputs_" + traj_name + "_dirt.txt";
  std::string states_gt_path        = data_folder + "/imu_states_gt_" + traj_name + ".txt";

  std::string error;
  read_states(states, states_gt_path, states_ext);

  if (inputs.size() != states.size())
    throw std::runtime_error("input and gt file size wrong!");
  if (inputs_dirt.size() != states.size())
    throw std::runtime_error("inputs_dirt and gt file size wrong!");
  if (inputs_bias.size() != states_ext.size()) {
    error = "inputs_bias and states_ext file size wrong!\n inputs_bias.size(): " + std::to_string(inputs_bias.size()) + " ; states_ext.size(): " + std::to_string(states_ext.size());
    throw std::runtime_error(error);
  }

  GpsMeasurements zs_gps;
  simulate_gps_data(zs_gps, states, gps_noise_std, gps_freq);

  // propagation input noise covariance: [ACC, GYRO, BIAS_ACC, BIAS_GYRO]
  Matrix12d Q = Matrix12d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
  Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);
  Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  //std::cout << "dimensione state: " << states.at(0).stateDim() << std::endl;
  //std::cout << "dimensione states_ext: " << states_ext.at(0).stateDim() << std::endl;

  // ------ Initializing the filter states to the first gt position
  NavStateExtended ukf_ext_state_dirt  = states_ext.at(0);
  NavStateExtended ukf_ext_state_bias  = states_ext.at(0);
  NavState         ukf_state           = states.at(0);
  NavState         ukf_state_bias      = states.at(0);
  NavState         ukf_state_dirt      = states.at(0);

  // ------- adding small heading error
  Eigen::AngleAxisd aa = Eigen::AngleAxisd(3.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ());
  ukf_ext_state_dirt.rotation()   = aa.toRotationMatrix() * ukf_ext_state_dirt.rotation();
  ukf_ext_state_dirt.position()  += Eigen::Vector3d::Constant(0.3);
  ukf_ext_state_bias.rotation()   = aa.toRotationMatrix() * ukf_ext_state_bias.rotation();
  ukf_ext_state_bias.position()  += Eigen::Vector3d::Constant(0.3);
  ukf_state.rotation()            = aa.toRotationMatrix() * ukf_state.rotation();
  ukf_state.position()           += Eigen::Vector3d::Constant(0.3);
  ukf_state_bias.rotation()       = aa.toRotationMatrix() * ukf_state_bias.rotation();
  ukf_state_bias.position()      += Eigen::Vector3d::Constant(0.3);
  ukf_state_dirt.rotation()       = aa.toRotationMatrix() * ukf_state_dirt.rotation();
  ukf_state_dirt.position()      += Eigen::Vector3d::Constant(0.3);
  
  // ------- initialize uncertainity
  ukf_ext_state_dirt.covariance() = Matrix15d::Identity();      // pose and velocity
  ukf_ext_state_dirt.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  ukf_ext_state_dirt.covariance().block<3, 3>(9, 9) *= 0.001;   // bias acc
  ukf_ext_state_dirt.covariance().block<3, 3>(12, 12) *= 0.001; // bias gyro
  ukf_ext_state_bias.covariance() = Matrix15d::Identity();      // pose and velocity
  ukf_ext_state_bias.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  ukf_ext_state_bias.covariance().block<3, 3>(9, 9) *= 0.001;   // bias acc
  ukf_ext_state_bias.covariance().block<3, 3>(12, 12) *= 0.001; // bias gyro
  ukf_state.covariance() = Matrix9d::Identity();      // pose and velocity
  ukf_state.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  ukf_state_bias.covariance() = Matrix9d::Identity();      // pose and velocity
  ukf_state_bias.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  ukf_state_dirt.covariance() = Matrix9d::Identity();      // pose and velocity
  ukf_state_dirt.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  
  // ------- UKF filter
  UKFImuExtendedGps ukf_ext_dirt, ukf_ext_bias;
  ukf_ext_dirt.setWeights(ukf_ext_state_dirt.stateDim(), 6, alpha);
  ukf_ext_bias.setWeights(ukf_ext_state_bias.stateDim(), 6, alpha);
  UKFImuGps ukf, ukf_bias, ukf_dirt;
  ukf.setWeights(ukf_state.stateDim(), 6, alpha);
  ukf_bias.setWeights(ukf_state_bias.stateDim(), 6, alpha);
  ukf_dirt.setWeights(ukf_state_dirt.stateDim(), 6, alpha);

  // ------- start from init gt state
  std::vector<NavStateExtended> estimates_ext_dirt(N), estimates_ext_bias(N);
  estimates_ext_dirt.at(0) = ukf_ext_state_dirt;
  estimates_ext_bias.at(0) = ukf_ext_state_bias;
  std::vector<NavState> estimates(N), estimates_bias(N), estimates_dirt(N);
  estimates.at(0) = ukf_state;
  estimates_bias.at(0) = ukf_state_bias;
  estimates_dirt.at(0) = ukf_state_dirt;

  GPSObs obs;
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * pow(gps_noise_std, 2);

  for (int i = 1; i < N; ++i) {
    // PROPAGATE with input covariance
    const Matrix6d Q_inputs = Q.block<6, 6>(0, 0);
    ukf_ext_dirt.propagate(ukf_ext_state_dirt, inputs_dirt.at(i - 1), Q_inputs, dt);
    ukf_ext_bias.propagate(ukf_ext_state_bias, inputs_bias.at(i - 1), Q_inputs, dt);
    ukf.propagate(ukf_state, inputs.at(i - 1), Q_inputs, dt);
    ukf_bias.propagate(ukf_state_bias, inputs_bias.at(i - 1), Q_inputs, dt);
    ukf_dirt.propagate(ukf_state_dirt, inputs_dirt.at(i - 1), Q_inputs, dt);

    // add bias covariance only to extended state
    const Matrix6d Q_bias = Q.block<6, 6>(6, 6);
    ukf_ext_state_dirt.covariance().block<6, 6>(9, 9) += Q_bias * dt * dt;
    ukf_ext_state_bias.covariance().block<6, 6>(9, 9) += Q_bias * dt * dt;

    if (zs_gps.count(i) > 0) {
      // here update
      const Eigen::Vector3d& curr_meas = zs_gps.at(i);
      obs.meas() = curr_meas;
      ukf_ext_dirt.update(ukf_ext_state_dirt, obs, R);
      ukf_ext_bias.update(ukf_ext_state_bias, obs, R);
      ukf.update(ukf_state, obs, R);
      ukf_bias.update(ukf_state_bias, obs, R);
      ukf_dirt.update(ukf_state_dirt, obs, R);
    }
    estimates_ext_dirt.at(i) = ukf_ext_state_dirt;
    estimates_ext_bias.at(i) = ukf_ext_state_bias;
    estimates.at(i) = ukf_state;
    estimates_bias.at(i) = ukf_state_bias;
    estimates_dirt.at(i) = ukf_state_dirt;
  }

  std::cerr << "\n                            i filtri hanno stimato le traiettorie con successo... " << std::endl;
  std::cout << "traiettoria: " << traj_name << std::endl;
  std::cout << "ampiezza traiettoria: " << radius << std::endl;
  std::cout << "durata traiettoria: " << T << " secondi" << std::endl;
  std::cout << "frequenza campionamento IMU: " << imu_freq << std::endl;
  std::cout << "frequenza campionamento GPS: " << gps_freq << std::endl;
  std::cout << "sigma ACC: " << imu_noise_std(0) << std::endl;
  std::cout << "sigma GYRO: " << imu_noise_std(1) << std::endl;
  std::cout << "sigma BIAS ACC: " << imu_noise_std(2) << std::endl;
  std::cout << "sigma BIAS GYRO: " << imu_noise_std(3) << std::endl;
  std::cout << "sigma GPS: " << gps_noise_std << std::endl;
  std::cout << std::endl;
  std::cout << " --- ATE  = absolute trajectory error without Umeyama Alignment\n";
  std::cout << " --- ATE2 = absolute trajectory error with Umeyama Alignment\n";
  std::cout << " --- RPE  = relative pose error with between 2 poses (poses can be consecutive or not, by default are consecutive)\n";
  std::cout << std::endl;
  std::cout << "NavState + Input " << std::endl;
  std::cout << " ATE              : " << computeATE(estimates, states) << std::endl;
  std::cout << " ATE2             : " << computeATE2(estimates, states) << std::endl;
  std::cout << " RPE translation  : " << computeRPET(estimates, states) << std::endl;
  std::cout << " RPE rotation     : " << computeRPER(estimates, states) << std::endl;
  std::cout << "NavState + Input_bias " << std::endl;
  std::cout << " ATE              : " << computeATE(estimates_bias, states) << std::endl;
  std::cout << " ATE2             : " << computeATE2(estimates_bias, states) << std::endl;
  std::cout << " RPE translation  : " << computeRPET(estimates_bias, states) << std::endl;
  std::cout << " RPE rotation     : " << computeRPER(estimates_bias, states) << std::endl;
  std::cout << "NavState + Input_dirt " << std::endl;
  std::cout << " ATE              : " << computeATE(estimates_dirt, states) << std::endl;
  std::cout << " ATE2             : " << computeATE2(estimates_dirt, states) << std::endl;
  std::cout << " RPE translation  : " << computeRPET(estimates_dirt, states) << std::endl;
  std::cout << " RPE rotation     : " << computeRPER(estimates_dirt, states) << std::endl;
  std::cout << "NavStateExtended + Input_bias " << std::endl;
  std::cout << " ATE              : " << computeATEext(estimates_ext_bias, states) << std::endl;
  std::cout << " ATE2             : " << computeATE2ext(estimates_ext_bias, states) << std::endl;
  std::cout << " RPE translation  : " << computeRPETExt(estimates_ext_bias, states) << std::endl;
  std::cout << " RPE rotation     : " << computeRPERExt(estimates_ext_bias, states) << std::endl;
  std::cout << "NavStateExtended + Input_dirt " << std::endl;
  std::cout << " ATE              : " << computeATEext(estimates_ext_dirt, states) << std::endl;
  std::cout << " ATE2             : " << computeATE2ext(estimates_ext_dirt, states) << std::endl;
  std::cout << " RPE translation  : " << computeRPETExt(estimates_ext_dirt, states) << std::endl;
  std::cout << " RPE rotation     : " << computeRPERExt(estimates_ext_dirt, states) << std::endl;

  

  {
    std::ofstream file(std::string(argv[1]) + "/traj_gt.txt");
    if (file.is_open()) {
      for (const auto& s : states) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est_ext_dirt.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_ext_dirt) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est_ext_bias.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_ext_bias) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est.txt");
    if (file.is_open()) {
      for (const auto& s : estimates) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est_bias.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_bias) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est_dirt.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_dirt) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/biases_est_dirt.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_ext_dirt) {
        const auto& b_acc = s.biasAcc();
        const auto& b_gyro = s.biasGyro();
        file << b_acc.x() << " " << b_acc.y() << " " << b_acc.z() << " " 
             << b_gyro.x() << " " << b_gyro.y() << " " << b_gyro.z() << "\n";
      }
      
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/biases_est_bias.txt");
    if (file.is_open()) {
      for (const auto& s : estimates_ext_bias) {
        const auto& b_acc = s.biasAcc();
        const auto& b_gyro = s.biasGyro();
        file << b_acc.x() << " " << b_acc.y() << " " << b_acc.z() << " " 
             << b_gyro.x() << " " << b_gyro.y() << " " << b_gyro.z() << "\n";
      }
      
      file.close();
    }
  }
}

void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, ImuInputs& inputs_bias_, ImuInputs& inputs_dirt_, 
                       const Vector4d& imu_noise_std_, const std::string& traj_name_, const std::string bias_gt_str,
                       const int T, const double radius, const int imu_freq, const int N, const double dt) {
                        
  Eigen::Vector3d g = Eigen::Vector3d(0.0, 0.0, -9.81);
  // TODO why negative
  std::string states_gt        = data_folder + "/imu_states_gt_" + traj_name_ + ".txt";
  std::string inputs_str       = data_folder + "/imu_inputs_" + traj_name_ + ".txt";
  std::string inputs_bias_str  = data_folder + "/imu_inputs_" + traj_name_ + "_bias.txt";
  std::string inputs_dirt_str  = data_folder + "/imu_inputs_" + traj_name_ + "_dirt.txt";
  std::string biases_str       = bias_gt_str + "/biases_gt.txt";
  std::string poses_str        = data_folder + "/poses_" + traj_name_ + ".txt";


  Eigen::ArrayXd times = -Eigen::ArrayXd::LinSpaced(N, 0.0, (N - 1) * dt);
  Eigen::MatrixXd poses = Eigen::MatrixXd::Zero(3, N);
  Eigen::ArrayXd poses_x(N);
  Eigen::ArrayXd poses_y(N);

  if (traj_name_ == "circular") {
    poses_x = radius * sin((times / T) * 2 * M_PI);
    poses_y = radius * cos((times / T) * 2 * M_PI);
  }
  else if (traj_name_ == "wave") {
    const auto r = radius * (times / T);
    poses_x = r;
    poses_y = (radius / 10) * cos((times / T) * 8 * M_PI);
  }
  else if (traj_name_ == "infinite") {
    poses_x = radius * sin((times / T) * 4 * M_PI);
    poses_y = radius * cos((times / T) * 2 * M_PI);
  }
  else if (traj_name_ == "spiral") {
    const auto r = radius * (times / T);
    poses_x = r * sin((times / T) * 4 * M_PI);
    poses_y = r * cos((times / T) * 4 * M_PI);
  }

  poses.row(0) = poses_x;
  poses.row(1) = poses_y;

  // obtaining vel and acc from positions
  Eigen::MatrixXd velocities    = Eigen::MatrixXd::Zero(3, N);
  Eigen::MatrixXd accelerations = Eigen::MatrixXd::Zero(3, N);

  for (int j = 1; j < poses.cols(); ++j) {
    velocities.col(j)    = poses.col(j) - poses.col(j - 1);
  }
  velocities    = velocities * 1.0 / dt;
  for (int j = 1; j < poses.cols(); ++j) {
    accelerations.col(j) = velocities.col(j) - velocities.col(j - 1);
  }
  accelerations = accelerations * 1.0 / dt;
  
  // ---------- Obtaining yaw from velocities  
  /*
  Eigen::ArrayXd yaw(N);
  Eigen::ArrayXd yaw_rate(N);
  for(int j = 0; j < N; ++j){
      double vx = velocities(0, j);
      double vy = velocities(1, j);
      yaw(j) = std::atan2(vy, vx);
  }
  // derivate of yaw → gyro z
  for(int j = 1; j < N; ++j){
      yaw_rate(j) = (yaw(j) - yaw(j-1)) / dt;
  }*/

  states_.resize(N);
  inputs_.resize(N);
  inputs_bias_.resize(N);
  inputs_dirt_.resize(N);

  // init state for GT computation
  states_.at(0).rotation() = Eigen::Matrix3d::Identity();
  states_.at(0).velocity() = velocities.col(0);
  states_.at(0).position() = poses.col(0);

  // ---------- Noise and Bias IMU
  Eigen::MatrixXd noises_gt = Eigen::MatrixXd::Zero(6, N);
  Eigen::MatrixXd biases_gt = Eigen::MatrixXd::Zero(6, N);

  for (int j = 1; j < poses.cols(); ++j) {

    const Eigen::Matrix3d& rot      = states_.at(j - 1).rotation();

    // get acceleration in imu-frame
    const Eigen::Vector3d& acc      = accelerations.col(j - 1);
    const Eigen::Vector3d t_acc     = rot.transpose() * (acc - g);

    Vector6d input    = Vector6d::Zero();
    input.head(3)     = t_acc;             // input   = [acc, gyro]
     // TODO: ...only acc at the moment...
    // input.tail(3) = gyro;               
    // input.tail(3) = rot.transpose() * [0.0, 0.0, yaw_rate(j-1)];

    // propagate state with zero noise for gt generation
    inputs_.at(j - 1)           = input;             // insert gt input
    inputs_bias_.at(j - 1)      = input;
    inputs_dirt_.at(j - 1)      = input;

    NavState state              = states_.at(j - 1);
    state.transition(input, Vector6d::Zero(), dt);
    states_.at(j) = state;

    // make input dirty :  INPUT = GT_INPUT + NOISE + BIAS 

    noises_gt.col(j).head(3) = imu_noise_std_(0) * Eigen::Vector3d::Random();// [noise acc]
    noises_gt.col(j).tail(3) = imu_noise_std_(1) * Eigen::Vector3d::Random();// [noise gyro]
    
    biases_gt.col(j).head(3) = biases_gt.col(j-1).head(3) + imu_noise_std_(2) * Eigen::Vector3d::Random(); //[bias acc]
    biases_gt.col(j).tail(3) = biases_gt.col(j-1).tail(3) + imu_noise_std_(3) * Eigen::Vector3d::Random(); //[bias gyro]
    //biases_gt.col(j).head(3) = Eigen::Vector3d::Constant(0.1);//   [bias acc]
    //biases_gt.col(j).tail(3) = Eigen::Vector3d::Constant(0.1);//   [bias gyro]

    inputs_.at(j - 1).head(3)       += noises_gt.col(j).head(3);
    inputs_.at(j - 1).tail(3)       += noises_gt.col(j).tail(3);
    inputs_bias_.at(j - 1).head(3)  += biases_gt.col(j).head(3);
    inputs_bias_.at(j - 1).tail(3)  += biases_gt.col(j).tail(3);
    inputs_dirt_.at(j - 1).head(3)  += noises_gt.col(j).head(3) + biases_gt.col(j).head(3);
    inputs_dirt_.at(j - 1).tail(3)  += noises_gt.col(j).tail(3) + biases_gt.col(j).tail(3);
  }


   // ===== SAVE STATES =====
  //   imu_states_gt_circular   imu_states_gt_wave   imu_states_gt_infinite   imu_states_gt_spiral
  std::ofstream state_file(states_gt);
  for (int i = 0; i < N-1; ++i) {
    const auto& rot_vec = lie_algebra::SO3Log(states_.at(i).rotation());
    const auto& p = states_.at(i).position();
    const auto& v = states_.at(i).velocity();
    state_file << dt << " "  //"dt rx ry rz vx vy vz \n" << 
               << rot_vec.x() << " " << rot_vec.y() << " " << rot_vec.z() << " " 
               << v.x() << " " << v.y() << " " << v.z() << " " 
               << p.x() << " " << p.y() << " " << p.z() << "\n";
  }
  state_file.close(); 
  /*
  //   imu_states_wave_ext   imu_states_infinite_ext   imu_states_spiral_ext
  std::ofstream state_ext_file(data_folder + "/imu_states_spiral_ext.txt");
  for (int i = 0; i < N-1; ++i) {
    const auto& rot_vec = lie_algebra::SO3Log(states_ext_.at(i).rotation());
    const auto& p       = states_ext_.at(i).position();
    const auto& v       = states_ext_.at(i).velocity();
    const auto& b_acc   = states_ext_.at(i).biasAcc();
    const auto& b_gyro  = states_ext_.at(i).biasGyro();
    const auto& rot_vec_2 = lie_algebra::SO3Log(states_.at(i).rotation());
    const auto& p_2       = states_.at(i).position();
    const auto& v_2       = states_.at(i).velocity();
    state_ext_file << dt << " "
               << rot_vec.x() -  rot_vec_2.x() << " " << rot_vec.y() - rot_vec_2.y() << " " << rot_vec.z() - rot_vec_2.z() << " " 
               << v.x() - v_2.x() << " " << v.y() - v_2.y() << " " << v.z() - v_2.z() << " " 
               << p.x() - p_2.x() << " " << p.y() - p_2.y() << " " << p.z() - p_2.z() << " "
               << b_acc.x() << " " << b_acc.y() << " " << b_acc.z() << " " 
               << b_gyro.x() << " " << b_gyro.y() << " " << b_gyro.z() << "\n";
  }
  state_ext_file.close();*/

/*
  // ===== SAVE STATES COVARIANCE =====
  // imu_states_circular   imu_states_wave   imu_states_infinite   imu_states_cov_spiral
  std::ofstream state_cov_file(data_folder + "/imu_states_cov_spiral.txt");
  for (int i = 0; i < N-1; ++i) {
    const auto& state_cov = states_.at(i).covariance();
    state_cov_file << dt << " "  
               << state_cov.row(0) << "\n" << state_cov.row(1) << "\n" << state_cov.row(2) << "\n" 
               << state_cov.row(3) << "\n" << state_cov.row(4) << "\n" << state_cov.row(5) << "\n" 
               << state_cov.row(6) << "\n" << state_cov.row(7) << "\n" << state_cov.row(8) << "\n\n";
  }
  state_cov_file.close();
  //   imu_states_wave_ext   imu_states_infinite_ext   imu_states_cov_spiral_ext
  std::ofstream state_cov_ext_file(data_folder + "/imu_states_cov_spiral_ext.txt");
  for (int i = 0; i < N-1; ++i) {
    const auto& state_cov = states_ext_.at(i).covariance();
    state_cov_ext_file << dt << " "  
               << state_cov.row(0) << "\n" << state_cov.row(1) << "\n" << state_cov.row(2) << "\n" 
               << state_cov.row(3) << "\n" << state_cov.row(4) << "\n" << state_cov.row(5) << "\n" 
               << state_cov.row(6) << "\n" << state_cov.row(7) << "\n" << state_cov.row(8) << "\n" 
               << state_cov.row(9) << "\n" << state_cov.row(10) << "\n" << state_cov.row(11) << "\n" 
               << state_cov.row(12) << "\n" << state_cov.row(13) << "\n" << state_cov.row(14) << "\n\n";
  }
  state_cov_ext_file.close();*/

   // ===== SAVE IMU INPUTS =====
  std::ofstream imu_file(inputs_str);
  for (int i = 0; i < N - 1; ++i) {
    const auto& u = inputs_.at(i);
    imu_file << dt << " " 
             << u(0) << " " << u(1) << " " << u(2) << " "   //  ax, ay, az
             << u(3) << " " << u(4) << " " << u(5) << "\n"; //  omega
  }
  imu_file.close();

  std::ofstream imu_bias_file(inputs_bias_str);
  for (int i = 0; i < N - 1; ++i) {
    const auto& u = inputs_bias_.at(i);
    imu_bias_file << dt << " " 
             << u(0) << " " << u(1) << " " << u(2) << " "   //  ax, ay, az
             << u(3) << " " << u(4) << " " << u(5) << "\n"; //  omega
  }
  imu_bias_file.close();

  std::ofstream imu_dirt_file(inputs_dirt_str);
  for (int i = 0; i < N - 1; ++i) {
    const auto& u = inputs_dirt_.at(i);
    imu_dirt_file << dt << " " 
             << u(0) << " " << u(1) << " " << u(2) << " "   //  ax, ay, az
             << u(3) << " " << u(4) << " " << u(5) << "\n"; //  omega
  }
  imu_dirt_file.close();


  // ===== SAVE IMU BIASES GT =====
  std::ofstream imu_unb_file(biases_str);
  //           imu_unb_file << "bias_acc[x, y, z]    bias_gyro[x, y, z]\n";
  for (int i = 0; i < N - 1; ++i) {
    const auto& b = biases_gt.col(i);
    imu_unb_file << b(3) << " " << b(4) << " " << b(5) << " "    //  bias acc
                 << b(0) << " " << b(1) << " " << b(2) << "\n";  //  bias gyro
  }
  imu_unb_file.close();
  
  // ===== SAVE poses, velocities, accelerations =====
  std::ofstream poses_file(poses_str);
  for(int j = 0; j < N; ++j){
      poses_file 
          << poses(0, j) << " "
          << poses(1, j) << " "
          << poses(2, j) << " "
          << velocities(0, j) << " "
          << velocities(1, j) << " "
          << velocities(2, j) << " "
          << accelerations(0, j) << " "
          << accelerations(1, j) << " "
          << accelerations(2, j) << "\n";
  }
  poses_file.close();


}

void simulate_gps_data(GpsMeasurements& zs_gps, const ImuStates& states, const double& gps_noise_std, const int gps_freq) {
  for (size_t n = 0; n < states.size(); ++n) {
    // id we are on the right freq then drop gps measurement
    if (n % gps_freq == 0) {
      const Eigen::Vector3d& gt_pose = states.at(n).position();
      Eigen::Vector3d noisy_gps      = gt_pose + gps_noise_std * Eigen::Vector3d::Random();
      zs_gps.insert(std::make_pair<int, Eigen::Vector3d>(std::forward<int>(n), std::forward<Eigen::Vector3d>(noisy_gps)));
    }
  }
}
