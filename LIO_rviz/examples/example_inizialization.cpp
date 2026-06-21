#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>
#include <ukf_manifold/lie_algebra.h>

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// -------------------------------------------- Utilities for usage: -----------------------------------------
//
// ./example_inizialization /home/giordano/Desktop/traj
//
// ./this-executable path-to-output
//
// ------------------------------------------- Utilities for plotting: ----------------------------------------
//
//  splot "gt_states.txt" using 1:2:3 w l, "estimated_0.txt" using 1:2:3 w l, "estimated_01.txt" using 1:2:3 w l, 
//        "estimated_02.txt" using 1:2:3 w l, "estimated_03.txt" using 1:2:3 w l, "estimated_gtAll.txt" using 1:2:3 w l
//        
//  splot "gt_states.txt" using 1:2:3 w l, "estimated_gtAll.txt" using 1:2:3 w l
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ukf_manifold;

const std::string data_folder(UKF_DATA_FOLDER);

using ImuStates    = std::vector<NavState>;
using ImuStatesExt = std::vector<NavStateExtended>;
using ImuInputs    = std::vector<Vector6d>;
using GpsMeasurements = std::map<int, Eigen::Vector3d>;
using IntegrationSteps = std::vector<double>; 

using Vector4d    = Eigen::Matrix<double, 4, 1>;
using Matrix12d   = Eigen::Matrix<double, 12, 12>;

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
double computeATE2ext(const ImuStatesExt& est, const ImuStatesExt& gt){
  assert(est.size() == gt.size());
  int N = est.size();
  Eigen::MatrixXd P_gt(3, N);
  Eigen::MatrixXd P_est(3, N);
  for(int i=0;i<N;i++){
    P_gt.col(i)  = gt[i].position();
    P_est.col(i) = est[i].position();
  }
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
double computeRPETExt(const ImuStatesExt& est, const ImuStatesExt& gt, int delta = 1){
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
double computeRPERExt(const ImuStatesExt& est, const ImuStatesExt& gt, int delta = 1){
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

NavStateExtended interpolate_state(const NavStateExtended& s1, const NavStateExtended& s2, double alpha){
  NavStateExtended s;
  s.position() = (1 - alpha) * s1.position() + alpha * s2.position(); // position
  s.velocity() = (1 - alpha) * s1.velocity() + alpha * s2.velocity(); // velocity
  Eigen::Quaterniond q1(s1.rotation());                               // rotation
  Eigen::Quaterniond q2(s2.rotation());
  Eigen::Quaterniond q = q1.slerp(alpha, q2);
  s.rotation() = q.toRotationMatrix();
  s.biasAcc()  = (1 - alpha) * s1.biasAcc()  + alpha * s2.biasAcc();  // biases
  s.biasGyro() = (1 - alpha) * s1.biasGyro() + alpha * s2.biasGyro();
  return s;
}

void interpolate_groundtruth(const std::vector<double>& t_imu, const std::vector<double>& t_gt,
                             const ImuStatesExt& gt_states, ImuStatesExt& gt_interp, std::vector<int>& imu_indices){
  int j = 0;
  int size = t_imu.size(), gtSize = t_gt.size();
  for (int i = 0; i < size; i++) {
    double t = t_imu[i];
    // find nearest timestamps between IMU and GT
    if (t < t_gt.front() || t > t_gt.back()) continue;  // TimeStamps of GT poses are included in TimeStamps of IMU 

    while (j + 1 < gtSize && t_gt[j+1] < t) j++;   // 
    if (j + 1 >= gtSize) break;
    // interpolate GT_STATE[TS_GT[j]] and GT_STATE[TS_GT[j+1]] because TS_GT[j] < TS_IMU[i] < TS_GT[j+1]
    double t1 = t_gt[j];
    double t2 = t_gt[j+1];
    double alpha = (t - t1) / (t2 - t1);
    NavStateExtended s_interp = interpolate_state(gt_states[j], gt_states[j+1], alpha);

    gt_interp.push_back(s_interp);
    imu_indices.push_back(i);
  }
}

void read_input(IntegrationSteps& dts_, ImuInputs& inputs_, const std::string& filename) {
  // ====== INPUTS IN EUROC dataset ---> [dt; w_x; w_y; w_z; a_x; a_y; a_z]
  // ====== INPUT IN UKF --------------> [a_x; a_y; a_z, ; w_x; w_y; w_z]

  std::ifstream file(filename.c_str());
  std::string line;
  std::getline(file, line);// Skip header

  while (std::getline(file, line)){
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(ss, cell, ',')) values.push_back(std::stod(cell));

    if (values.size() >= 7){
      inputs_.emplace_back(values[4], values[5], values[6], values[1], values[2], values[3]);
      double timestamp = values[0] * 1e-9;  // from Nanoseconds to Seconds
      dts_.push_back(timestamp);
    }
  }
  inputs_.shrink_to_fit();
  dts_.shrink_to_fit();
}

void read_states(IntegrationSteps& dts_, ImuStatesExt& states, const std::string filename) {
  std::ifstream file(filename.c_str());
  std::string line; 
  std::getline(file, line);// Skip header
  int i = 0;
  while (std::getline(file, line)){
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(ss, cell, ',')) values.push_back(std::stod(cell));

    if (values.size() >= 17){
      // dataset:  dt, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bw_x, bw_y, bw_z, ba_x, ba_y, ba_z;
      Eigen::Quaterniond quat(values[4], values[5], values[6], values[7]);
      quat.normalize();
      const Eigen::Matrix3d R       = quat.toRotationMatrix();
      const Eigen::Vector3d v       = Eigen::Vector3d(values[8], values[9], values[10]);
      const Eigen::Vector3d p       = Eigen::Vector3d(values[1], values[2], values[3]);
      const Eigen::Vector3d b_gyro  = Eigen::Vector3d(values[11], values[12], values[13]);
      const Eigen::Vector3d b_acc   = Eigen::Vector3d(values[14], values[15], values[16]);

      NavStateExtended state;
      state.rotation() = R;
      state.velocity() = v;
      state.position() = p;
      state.biasAcc() = b_acc;
      state.biasGyro() = b_gyro;

      states.push_back(state);
      double timestamp = values[0] * 1e-9;
      dts_.push_back(timestamp);
    }
    else{
      std::cout << "line " << i << " not inizialized: " << line << std::endl;
    }
    ++i;
  }
  states.shrink_to_fit();
  dts_.shrink_to_fit();
}

void simulate_gps_data(GpsMeasurements& gps_input, const ImuStatesExt& states, const double& gps_noise_std, const int gps_freq) {
  for (size_t n = 0; n < states.size(); ++n) {
    // if we are on the right freq then drop gps measurement
    if (n % gps_freq == 0) {
      const Eigen::Vector3d& gt_pose = states.at(n).position();
      Eigen::Vector3d noisy_gps      = gt_pose + gps_noise_std * Eigen::Vector3d::Random();
      gps_input.insert(std::make_pair<int, Eigen::Vector3d>(std::forward<int>(n), std::forward<Eigen::Vector3d>(noisy_gps)));
    }
  }
}

void initialization_1(Eigen::Matrix<double,15,15>& P0, Eigen::Matrix3d& R0, Eigen::Vector3d& ba0, 
                      Eigen::Vector3d& bg0, int INIT_STEPS, ImuInputs& inputs, std::vector<int>& imu_indices){
  // (ASSUMPTION: system is not moving for the first frames)
  std::cout << "\n==========  START ==========\n";
  std::cout << " INIT_STEPS=" << INIT_STEPS
            << "  imu_indices.size()=" << imu_indices.size()
            << "  inputs.size()=" << inputs.size() << "\n";
  Vector6d mean = Vector6d::Zero();
  for (int i = 0; i < INIT_STEPS; ++i){  
    Vector6d init_input = inputs[imu_indices[i]];
    mean += init_input;
  }
  mean /= INIT_STEPS;
  
  Eigen::Vector3d gyro_mean = mean.tail(3);
  Eigen::Vector3d acc_mean  = mean.head(3);
  // if |acc_mean.norm() - 9.81| < 0.01  &&  gyro_mean.norm() < 0.01  ===> the system is not moving
  // otherwise we have to decrease INIT_STEPS and rechek;
  Eigen::Vector3d g_meas = acc_mean.normalized();  // acc measures orientation
  Eigen::Vector3d g_world(0.0, 0.0, -1.0);         // gravity orientation
  Eigen::Vector3d v = g_meas.cross(g_world);       // acc (vec) gravity 
  double c = g_meas.dot(g_world);                  // acc (dot) gravity 
  double s = v.norm();
  R0 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d vx;
  if (s > 1e-8){
    vx <<     0, -v.z(),  v.y(),
            v.z(),     0, -v.x(),
          -v.y(),  v.x(),     0;
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1 - c) / (s * s)); // rotation from IMU to World frame
  }
  g_world *= 9.81;   // give magnitude to gravity
  ba0 = acc_mean - R0.transpose() * g_world;
  bg0 = gyro_mean;

  P0(0,0) = 1e-3;     // Orientation: roll  small  
  P0(1,1) = 1e-3;     //              pitch small
  P0(2,2) = 0.1;      //              yaw   large
  P0.block<3,3>(3,3) = 1e-4 * Eigen::Matrix3d::Identity();   // Position
  P0.block<3,3>(6,6) = 0.01 * Eigen::Matrix3d::Identity();   // Velocity
  P0.block<3,3>(9,9) = 1e-3 * Eigen::Matrix3d::Identity();   // Acc bias
  P0.block<3,3>(12,12) = 1e-4 * Eigen::Matrix3d::Identity(); // Gyro bias

  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << R0(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << " pos0:   " << Eigen::Vector3d::Zero().transpose()  << "\n";
  std::cout << " vel0:   " << Eigen::Vector3d::Zero().transpose()  << "\n";
  std::cout << " ba0:    " << ba0.transpose()   << "\n";
  std::cout << " bg0:    " << bg0.transpose()   << "\n";
  std::cout << " P0 diag: " << P0.diagonal().transpose() << "\n";
  std::cout << " g_est:  " << acc_mean.transpose() << "  |g|=" << acc_mean.norm() << "\n";
  std::cout << "==========  END ==========\n\n";
}

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

void initialization_2(Eigen::Matrix<double,15,15>& P0,
                      Eigen::Matrix3d&   R0,
                      Eigen::Vector3d&   pos0,
                      Eigen::Vector3d&   vel0,
                      Eigen::Vector3d&   ba0,
                      Eigen::Vector3d&   bg0,
                      int                INIT_STEPS,
                      ImuInputs&         inputs,
                      std::vector<int>&  imu_indices,
                      GpsMeasurements&   gps_input,
                      const std::vector<double>& dts_imu,
                      double& gps_noise_std,
                      Vector4d imu_noise_std){
  std::cout << "\n========== [init2] START ==========\n";
  std::cout << " INIT_STEPS=" << INIT_STEPS
            << "  imu_indices.size()=" << imu_indices.size()
            << "  dts_imu.size()=" << dts_imu.size()
            << "  inputs.size()=" << inputs.size()
            << "  gps_input.size()=" << gps_input.size() << "\n";

  std::vector<int>             gps_indices; 
  std::vector<Eigen::Vector3d> p_gps;
  std::vector<double>          t_gps;
  for (int k = 0; k < INIT_STEPS; ++k) {
    if (k >= (int)imu_indices.size()) {
      std::cerr << " ERRORE: k=" << k << " >= imu_indices.size()="
                << imu_indices.size() << "\n";
      break;
    }
    int imu_idx = imu_indices[k];
    if (imu_idx >= (int)dts_imu.size()) {
      std::cerr << " ERRORE: imu_indices[" << k << "]=" << imu_idx
                << " >= dts_imu.size()=" << dts_imu.size() << "\n";
      break;
    }
    if (gps_input.count(k)) {
      p_gps.push_back(gps_input[k]);
      t_gps.push_back(dts_imu[imu_indices[k]]);
      gps_indices.push_back(imu_indices[k]);
    }
  }

  const int N = (int)p_gps.size() - 1;
  
  if (N < 3) {
    std::cerr << " Troppi pochi GPS fix (serve N>=3), fallback a zero.\n";
    P0 = Eigen::Matrix<double,15,15>::Identity() * 0.1;
    R0 = Eigen::Matrix3d::Identity();
    pos0 = vel0 = ba0 = bg0 = Eigen::Vector3d::Zero();
    return;
  }
  //  Stima grezza del gyro bias: Se non è fermo, si può mettere bg0 = Zero e lasciare al filtro.  
  Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  int cnt = 0;
  int bg_steps = std::min(INIT_STEPS/4, (int)inputs.size());
  for (int k = 0; k < bg_steps; ++k) {
    if (imu_indices[k] >= (int)inputs.size()) {
      std::cerr << " ERRORE bg loop: imu_indices[" << k << "]="
                << imu_indices[k] << " >= inputs.size()=" << inputs.size() << "\n";
      break;
    }
    gyro_sum += inputs[imu_indices[k]].tail<3>();
    ++cnt;
  }
  bg0 = (cnt > 0) ? (gyro_sum / cnt) : zero;
    
  //Preintegrazione di ogni segmento GPS_i -> GPS_{i+1}  con ba = 0, bg = bg0 stimato
  std::vector<PreintSegment> segs;
  segs.reserve(N);
  const Eigen::Vector3d ba_init = Eigen::Vector3d::Zero();

  for (int i = 0; i < N; ++i) {
    segs.push_back(preintegrate_segment(gps_indices[i], gps_indices[i+1], inputs, dts_imu, bg0, ba_init));
  }
 
  // ------ Sistema lineare  A * x = b ------                   Δp_GPS - Rì*Δp_preint = function(preintegrations_deltas)
  //   
  //  p_{i+1} - p_i  =  v_i * dt_i  +  R_i * Δp_i  +  0.5 * g * dt_i² - R_i * J_p_ba * ba
  //
  //  Riscriviamo come: p_{i+1} - p_i - R_i * Δp_i  =  v_i * dt_i + 0.5 * dt_i² * g - R_i * J_p_ba * ba
  //
  //  Incognite:  x = [v_0(3), ..., v_{N-1}(3),  g(3),  ba(3)]
  
  const int STATE_SIZE = 3 * N + 6;   // [v_0..v_{N-1}, g, ba]
  std::cout << " sistema lineare: " << 3*N << " equazioni, " << STATE_SIZE << " incognite" 
            << " (sottodeterminato di 6 -> ba stimato in norma minima)\n";
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * N, STATE_SIZE);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * N);
 
  for (int i = 0; i < N; ++i){
    const double dt = t_gps[i+1] - t_gps[i];
    // Rotazione assoluta al keyframe i: al primo passo assumiamo R_i = Identity (verrà corretta dopo).
    const Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();

    A.block<3,3>(3*i, 3*i)   = dt * Eigen::Matrix3d::Identity();
    A.block<3,3>(3*i, 3*N)   = 0.5 * dt * dt * Eigen::Matrix3d::Identity();
    A.block<3,3>(3*i, 3*N+3) = -Ri * segs[i].J_p_ba;

    b.segment<3>(3*i) = p_gps[i+1] - p_gps[i] - Ri * segs[i].delta_p;
  }
  // Risoluzione least-squares (sistema sovradeterminato se N > 2)
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
 
  vel0 = x.segment<3>(3 * (N-1));                // v_N velocità all'ultimo keyframe GPS usato per l'inizializzazione, nel world frame
  Eigen::Vector3d g_est = x.segment<3>(3*N);     // gravità stimata nel world frame
  ba0                   = x.segment<3>(3*N + 3); // accelerometer bias
 
  // Normalizza la gravità al valore fisico
  if (g_est.norm() > 1e-3)
      g_est = g_est.normalized() * 9.81;
  else
      g_est = Eigen::Vector3d(0.0, 0.0, -9.81);  // fallback

  // Rotation from IMU frame to World frame
  Eigen::Vector3d g_meas  = g_est.normalized();     // direzione g stimata
  Eigen::Vector3d g_world(0.0, 0.0, -1.0);          // g atteso nel world (Z-up)
  Eigen::Vector3d v_cross = g_meas.cross(g_world);
  double          c       = g_meas.dot(g_world);
  double          s_norm  = v_cross.norm();

  R0 = Eigen::Matrix3d::Identity();          // calcolo orientamento iniziale dell'IMU
  if (s_norm > 1e-8) {
    Eigen::Matrix3d vx = lie_algebra::skew(v_cross);
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1.0 - c) / (s_norm * s_norm));
  }
  Eigen::Matrix3d R_T = R0;                  // Rotazione dell'IMU frame con le rotazioni della preintegrazione
  for (int i = 0; i < N; ++i) 
    R_T = R_T * segs[i].delta_R;
  R0 = R_T;

  // Posizione iniziale = ultima posizione GPS dell'inizializzazione
  pos0 = p_gps[N];

  //  Stimiamo la qualità della soluzione dai residui del sistema Ax=b.
  Eigen::VectorXd residuals = A * x - b;
  double sigma2_res = residuals.squaredNorm() / std::max(3*N - STATE_SIZE, 1);

  P0.setZero();
  // Rotazione: incertezza derivata dalla direzione di g
  // (yaw non osservabile da GPS solo → grande incertezza)
  P0(0,0) = 1e-3;   // roll
  P0(1,1) = 1e-3;   // pitch
  P0(2,2) = 0.5;    // yaw (non osservabile)

  // Posizione: rumore GPS
  P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();

  // Velocità: incertezza dalla soluzione LS
  P0.block<3,3>(6,6)   = std::max(sigma2_res, 0.01) * Eigen::Matrix3d::Identity();

  double T_init = t_gps.back() - t_gps.front();  // durata finestra init [s]
  P0.block<3,3>(9,9)   = pow(imu_noise_std(2), 2) * T_init * Eigen::Matrix3d::Identity();  // bias accel: sigma_ba² * T_init
  P0.block<3,3>(12,12) = pow(imu_noise_std(3), 2) * T_init * Eigen::Matrix3d::Identity();  // bias gyro : sigma_bg² * T_init

  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << R0(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << " pos0:   " << pos0.transpose()  << "\n";
  std::cout << " vel0:   " << vel0.transpose()  << "\n";
  std::cout << " ba0:    " << ba0.transpose()   << "\n";
  std::cout << " bg0:    " << bg0.transpose()   << "\n";
  std::cout << " P0 diag: " << P0.diagonal().transpose() << "\n";
  std::cout << " g_est:  " << g_est.transpose() << "  |g|=" << g_est.norm() << "\n";
  std::cout << " T_init: " << T_init << " s\n";
  std::cout << "==========  END ==========\n\n";
}

void initialization_3(Eigen::Matrix<double,15,15>& P0,
                      Eigen::Matrix3d&   R0,
                      Eigen::Vector3d&   pos0,
                      Eigen::Vector3d&   vel0,
                      Eigen::Vector3d&   ba0,
                      Eigen::Vector3d&   bg0,
                      int                STATIC_STEPS,
                      int                MOTION_STEPS,
                      ImuInputs&         inputs,
                      std::vector<int>&  imu_indices,
                      GpsMeasurements&   gps_input,
                      const std::vector<double>& dts_imu,
                      double             gps_noise_std,
                      Vector4d           imu_noise_std){
      // mix tra le due inizializzazioni
  // ASSUNZIONE: robot fermo nei primi frames, stima R0 / ba0 / bg0
  std::cout << "\n========== [init3] START ==========\n";
  std::cout << "  STATIC_STEPS=" << STATIC_STEPS
            << "  MOTION_STEPS=" << MOTION_STEPS
            << "  imu_indices.size()=" << imu_indices.size()
            << "  dts_imu.size()=" << dts_imu.size()
            << "  inputs.size()=" << inputs.size()
            << "  gps_input.size()=" << gps_input.size() << "\n";
  P0.setZero();
  Vector6d mean = Vector6d::Zero();
  for (int i = 0; i < STATIC_STEPS; ++i) mean += inputs[imu_indices[i]];
  mean /= STATIC_STEPS;

  const Eigen::Vector3d gyro_mean = mean.tail<3>();
  const Eigen::Vector3d acc_mean  = mean.head<3>();

  // Rotazione iniziale: allinea il body frame alla gravità
  Eigen::Vector3d g_meas  = acc_mean.normalized();
  Eigen::Vector3d g_world(0.0, 0.0, -1.0);
  Eigen::Vector3d v_cross = g_meas.cross(g_world);
  double c = g_meas.dot(g_world), s = v_cross.norm();
  R0 = Eigen::Matrix3d::Identity();
  if (s > 1e-8) {
    Eigen::Matrix3d vx = lie_algebra::skew(v_cross);
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1.0 - c) / (s * s));
  }
  Eigen::Vector3d g_world_full(0.0, 0.0, -9.81);
  ba0 = acc_mean - R0.transpose() * g_world_full;
  bg0 = gyro_mean;

  // P0: blocchi rotazione e bias (ben stimati da init_1)
  P0(0,0) = 1e-3;   // roll  (ben osservabile da g)
  P0(1,1) = 1e-3;   // pitch (ben osservabile da g)
  P0(2,2) = 0.5;    // yaw   (non osservabile senza magnetometro)
  P0.block<3,3>(9,9)   = pow(imu_noise_std(2), 2) * Eigen::Matrix3d::Identity();  // ba
  P0.block<3,3>(12,12) = pow(imu_noise_std(3), 2) * Eigen::Matrix3d::Identity();  // bg
  
  const Eigen::Vector3d g_known(0.0, 0.0, -9.81);
  
  std::vector<int>             gps_indices;
  std::vector<Eigen::Vector3d> p_gps;
  std::vector<double>          t_gps;
 
  for (int k = STATIC_STEPS; k < MOTION_STEPS; ++k) {
    if (k >= (int)imu_indices.size()) break;
    int imu_idx = imu_indices[k];
    if (imu_idx >= (int)dts_imu.size()) break;
    if (gps_input.count(k)) {
      p_gps.push_back(gps_input[k]);
      t_gps.push_back(dts_imu[imu_idx]);
      gps_indices.push_back(imu_idx);
    }
  }
 
  const int N = (int)p_gps.size() - 1;
  std::cout << " N segmenti GPS=" << N << "\n";
 
  if (N < 2) {
    // Non abbastanza GPS: teniamo solo i risultati di init_1
    // vel0 e pos0 rimangono zero — il filtro li stimerà
    std::cerr << " GPS insufficienti (N=" << N
              << "), vel0 e pos0 impostati a zero.\n";
    pos0 = Eigen::Vector3d::Zero();
    vel0 = Eigen::Vector3d::Zero();
    P0.block<3,3>(3,3) = 10.0 * Eigen::Matrix3d::Identity();  // pos: alta incertezza
    P0.block<3,3>(6,6) = 10.0 * Eigen::Matrix3d::Identity();  // vel: alta incertezza
    return;
  }
 
  // --- preintegrazione con ba0 e bg0 da init_1 ---
  std::vector<PreintSegment> segs;
  segs.reserve(N);
  for (int i = 0; i < N; ++i)
    segs.push_back(preintegrate_segment(gps_indices[i], gps_indices[i+1], inputs, dts_imu, bg0, ba0));  // ← usa i bias già stimati da init_1
 
  // --- sistema lineare con g FISSA ---
  //
  //  Modello per segmento i: p_{i+1} - p_i = v_i*dt + R_i*Δp_i + 0.5*g*dt² - R_i*J_p_ba*Δba
  //
  //  Con g nota, portiamo 0.5*g*dt² a destra (termine noto).
  //  Incognite: x = [v_0(3), ..., v_{N-1}(3), Δba(3)]
  //             dimensione: 3N + 3
  //
  //  Nota: Δba è la *correzione* al ba già stimato da init_1.
  //  Il sistema è 3N equazioni e 3N+3 incognite:
  //    - sovradeterminato se N > 1 per le velocità
  //    - Δba osservabile se i segmenti hanno accelerazioni diverse
  //
  const int STATE_SIZE = 3 * N + 3;   // [v_0..v_{N-1}, Δba]
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*N, STATE_SIZE);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(3*N);
 
  for (int i = 0; i < N; ++i) {
    const double dt   = t_gps[i+1] - t_gps[i];
    Eigen::Matrix3d Ri = R0;                                // R0 = R-IMU iniziale
    for (int j = 0; j < i; ++j) Ri = Ri * segs[j].delta_R;  // Ri 
 
    A.block<3,3>(3*i, 3*i) = dt * Eigen::Matrix3d::Identity();
    A.block<3,3>(3*i, 3*N) = -Ri * segs[i].J_p_ba;
 
    // Termine noto: GPS diff - preintegrazione nominale - contributo gravità
    b.segment<3>(3*i) = p_gps[i+1] - p_gps[i] - Ri * segs[i].delta_p - 0.5 * dt * dt * g_known;
  }
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  vel0 = x.segment<3>(3 * (N-1)); 
  
  Eigen::Vector3d delta_ba = x.segment<3>(3*N);   // Correzione bias accel (Δba = correzione su ba0 già stimato da init_1)
  ba0 += delta_ba;                                // raffinamento del bias accel
  pos0 = p_gps[N];                                // Posizione = ultima posizione GPS della finestra
  Eigen::Matrix3d R_T = R0;                       // Propaga R0 fino al timestamp T con le rotazioni preintegrate
  for (int i = 0; i < N; ++i) R_T = R_T * segs[i].delta_R;
  R0 = R_T;

  // Cerca la prima coppia GPS con spostamento significativo
  Eigen::Vector3d heading = Eigen::Vector3d::Zero();
  for (int i = 0; i + 1 < (int)p_gps.size(); ++i) {
    Eigen::Vector3d dp = p_gps[i+1] - p_gps[i];
    dp.z() = 0.0;
    if (dp.norm() > 1.0) {  // spostamento > 2*sigma_GPS
      heading = dp.normalized();
      std::cout << " heading GPS trovato a segmento " << i 
                << "  |dp|=" << dp.norm() << "\n";
      break;
    }
  }
  if (heading.norm() > 0.5) {
      double yaw_gps = std::atan2(heading.y(), heading.x());
      // Decomposizione di R0 in roll/pitch (da gravità) + sostituzione yaw (da GPS)
      Eigen::Vector3d euler = R0.eulerAngles(2, 1, 0);  // ZYX: yaw, pitch, roll
      
      R0 = Eigen::AngleAxisd(yaw_gps,    Eigen::Vector3d::UnitZ()).toRotationMatrix()
          * Eigen::AngleAxisd(euler(1),   Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(euler(2),   Eigen::Vector3d::UnitX()).toRotationMatrix();
      
      std::cout << " yaw da GPS=" << yaw_gps * 180.0/M_PI << "°"
                << "  yaw precedente=" << euler(0) * 180.0/M_PI << "°\n";
  } else {
      std::cout << " heading GPS non affidabile, yaw non corretto\n";
  }
 
  // --- qualità della soluzione ---
  Eigen::VectorXd residuals = A * x - b;
  double sigma2_res = residuals.squaredNorm() / std::max(N, 1);
 
  // P0: blocchi posizione e velocità (da init_2)
  P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();
  P0.block<3,3>(6,6) = std::max(sigma2_res, 0.01) * Eigen::Matrix3d::Identity();
 
  // Raffina P0 dei bias con T_init (come discusso in precedenza)
  double T_init = t_gps.back() - t_gps.front();
  // se Δba è grande, P ba cresce
  P0.block<3,3>(9,9)   = std::max(pow(imu_noise_std(2), 2) * T_init, delta_ba.squaredNorm() / 3.0) * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = pow(imu_noise_std(1), 2) * T_init * Eigen::Matrix3d::Identity();
 
  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << R0(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << " pos0="  << pos0.transpose()  << "\n";
  std::cout << " vel0="  << vel0.transpose()  << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose()   << "\n";
  std::cout << " P0 diag=" << P0.diagonal().transpose() << "\n";
  std::cout << " g_est:  " << acc_mean.transpose() << "  |g|=" << acc_mean.norm() << "\n";
  std::cout << " T_init=" << T_init << " s\n";
  std::cout << "==========  END ==========\n\n";
}

void initialization_4(Eigen::Matrix<double,15,15>& P0,
                      Eigen::Matrix3d&   R0,
                      Eigen::Vector3d&   pos0,
                      Eigen::Vector3d&   vel0,
                      Eigen::Vector3d&   ba0,
                      Eigen::Vector3d&   bg0,
                      int                MAX_STEPS,       // massimo campioni da esaminare
                      ImuInputs&         inputs,
                      std::vector<int>&  imu_indices,
                      GpsMeasurements&   gps_input,
                      const std::vector<double>& dts_imu,
                      double             gps_noise_std,
                      Vector4d           imu_noise_std,
                      double             thr = 0.08,  // [m²] soglia varianza GPS
                      double             thr_acc = 0.08){  // [m/s²] soglia acc non-grav
  P0.setZero();
  thr = 2.0 * gps_noise_std * gps_noise_std;  // = 0.5 m²
  std::cout << "\n========== [init4] START ==========\n";

  struct GpsFrame {
    int    k;        // indice in imu_indices
    int    imu_idx;  // indice in dts_imu / inputs
    Eigen::Vector3d pos;
    double t;
  };
  std::vector<GpsFrame> gps_frames;
 
  for (int k = 0; k < MAX_STEPS && k < (int)imu_indices.size(); ++k) {
    int imu_idx = imu_indices[k];
    if (imu_idx >= (int)dts_imu.size()) break;
    if (gps_input.count(k)) gps_frames.push_back({k, imu_idx, gps_input[k], dts_imu[imu_idx]});
  }
 
  if (gps_frames.size() < 2) {
    std::cerr << " GPS insufficienti, fallback a init_1 pura.\n";
    initialization_1(P0, R0, ba0, bg0, MAX_STEPS, inputs, imu_indices);
    pos0 = vel0 = Eigen::Vector3d::Zero();
    P0.block<3,3>(3,3) = 10.0 * Eigen::Matrix3d::Identity();
    P0.block<3,3>(6,6) = 10.0 * Eigen::Matrix3d::Identity();
    return;
  }
  //  Rilevamento dinamico: Sistema in Moto o Sistema in Quiete
  int static_end_k = gps_frames.back().k;  // default: tutto statico
  bool motion_detected = false;
  for (int fi = 1; fi < (int)gps_frames.size(); ++fi) {
    // varianza posizione GPS su finestra [0..fi] 
    Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
    for (int j = 0; j <= fi; ++j) mean_pos += gps_frames[j].pos;
    mean_pos /= (fi + 1);
    double var = 0.0;
    for (int j = 0; j <= fi; ++j)
      var += (gps_frames[j].pos - mean_pos).squaredNorm();
    var /= (fi + 1);
    // modulo accelerazioni misurate
    Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero();
    int acc_cnt = 0;
    for (int k = gps_frames[fi-1].imu_idx; k < gps_frames[fi].imu_idx &&
         k < (int)inputs.size(); ++k) {
      acc_sum += inputs[k].head<3>();
      ++acc_cnt;
    }
    double a_nongrav = 0.0;
    Eigen::Vector3d acc_mean;
    if (acc_cnt > 0) {
      acc_mean = acc_sum / acc_cnt;
      a_nongrav = std::abs(acc_mean.norm() - 9.81);
    }
    std::cout << "  GPS frame " << fi
              << "  var=" << var
              << "  a_nongrav=" << a_nongrav << "\n";
 
    // OR: basta uno dei due per dichiarare moto
    if (var > thr || a_nongrav > thr_acc) {
      static_end_k = gps_frames[fi-1].k;  // l'ultimo frame ancora statico
      motion_detected = true;
      std::cout << " MOTO rilevato al GPS frame " << fi
                << "  static_end_k=" << static_end_k << "\n";
      break;
    }
  }
 
  if (!motion_detected) std::cout << " nessun moto rilevato nella finestra, tutto trattato come statico.\n";
  // se tutta finestra statica → R0, ba, bg
  //int static_imu_end = imu_indices[static_end_k];
  Eigen::Vector3d acc_sum_s  = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_sum_s = Eigen::Vector3d::Zero();
  int cnt_s = 0;
  for (int k = 0; k <= static_end_k && k < (int)imu_indices.size(); ++k) {
    int idx = imu_indices[k];
    if (idx >= (int)inputs.size()) break;
    acc_sum_s  += inputs[idx].head<3>();
    gyro_sum_s += inputs[idx].tail<3>();
    ++cnt_s;
  }
 
  std::cout << " finestra statica: " << cnt_s << " campioni IMU\n";
 
  if (cnt_s < 10) {
    std::cerr << " finestra statica troppo corta, uso media globale.\n";
    for (int k = 0; k < MAX_STEPS && k < (int)imu_indices.size(); ++k) {
      int idx = imu_indices[k];
      if (idx >= (int)inputs.size()) break;
      acc_sum_s  += inputs[idx].head<3>();
      gyro_sum_s += inputs[idx].tail<3>();
      ++cnt_s;
    }
  }
 
  Eigen::Vector3d acc_mean_s  = acc_sum_s  / cnt_s;
  Eigen::Vector3d gyro_mean_s = gyro_sum_s / cnt_s;
 
  // Rotazione da allineamento gravità
  Eigen::Vector3d g_meas  = acc_mean_s.normalized();
  Eigen::Vector3d g_world_dir(0.0, 0.0, -1.0);
  Eigen::Vector3d vc = g_meas.cross(g_world_dir);
  double c_dot = g_meas.dot(g_world_dir), s_vc = vc.norm();
  R0 = Eigen::Matrix3d::Identity();
  if (s_vc > 1e-8) {
    Eigen::Matrix3d vx = lie_algebra::skew(vc);
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1.0 - c_dot) / (s_vc * s_vc));
  }
  Eigen::Vector3d g_world_full(0.0, 0.0, -9.81);
  ba0 = acc_mean_s - R0.transpose() * g_world_full;
  bg0 = gyro_mean_s;
 
  std::cout << " R0 det=" << R0.determinant()
            << "  |g_meas|=" << acc_mean_s.norm() << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose() << "\n";
 
  // P0 blocchi rotazione e bias dalla finestra statica
  P0(0,0) = 1e-3; P0(1,1) = 1e-3; P0(2,2) = 0.5;
  P0.block<3,3>(9,9)   = pow(imu_noise_std(2), 2) * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = pow(imu_noise_std(3), 2) * Eigen::Matrix3d::Identity();
 
  // finestra dinamica: raccoglie tutti i GPS frame da static_end_k in poi
  std::vector<int>             gps_indices_m;
  std::vector<Eigen::Vector3d> p_gps_m;
  std::vector<double>          t_gps_m;
 
  for (auto& gf : gps_frames) {
    if (gf.k >= static_end_k) {  // incluso l'ultimo statico come punto di partenza
      gps_indices_m.push_back(gf.imu_idx);
      p_gps_m.push_back(gf.pos);
      t_gps_m.push_back(gf.t);
    }
  }
 
  const int N_m = (int)p_gps_m.size() - 1;
  std::cout << " finestra moto: " << N_m << " segmenti GPS\n";
 
  if (N_m < 1) {
    std::cerr << " nessun segmento in moto, vel0=0, pos0=primo GPS.\n";
    vel0 = Eigen::Vector3d::Zero();
    pos0 = gps_frames.back().pos;
    P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();
    P0.block<3,3>(6,6) = 10.0 * Eigen::Matrix3d::Identity();
    return;
  }
 
  // Preintegrazione segmenti in moto con bias da finestra statica
  std::vector<PreintSegment> segs_m;
  segs_m.reserve(N_m);
  for (int i = 0; i < N_m; ++i)
    segs_m.push_back(preintegrate_segment(
        gps_indices_m[i], gps_indices_m[i+1],
        inputs, dts_imu, bg0, ba0));
 
  // Sistema lineare con g fissa: x = [v_0..v_{N-1}, Δba]
  const Eigen::Vector3d g_known(0.0, 0.0, -9.81);
  const int STATE_SIZE_M = 3 * N_m + 3;
  Eigen::MatrixXd A_m = Eigen::MatrixXd::Zero(3 * N_m, STATE_SIZE_M);
  Eigen::VectorXd b_m = Eigen::VectorXd::Zero(3 * N_m);
 
  for (int i = 0; i < N_m; ++i) {
    const double dt = t_gps_m[i+1] - t_gps_m[i];
    Eigen::Matrix3d Ri = R0;
    for (int j = 0; j < i; ++j) Ri = Ri * segs_m[j].delta_R;
 
    A_m.block<3,3>(3*i, 3*i) = dt * Eigen::Matrix3d::Identity();
    A_m.block<3,3>(3*i, 3*N_m) = -Ri * segs_m[i].J_p_ba;
    b_m.segment<3>(3*i) = p_gps_m[i+1] - p_gps_m[i]
                          - Ri * segs_m[i].delta_p
                          - 0.5 * dt * dt * g_known;
  }
 
  Eigen::VectorXd x_m = A_m.colPivHouseholderQr().solve(b_m);
 
  vel0 = x_m.segment<3>(3 * (N_m - 1));
  Eigen::Vector3d delta_ba = x_m.segment<3>(3 * N_m);
  ba0 += delta_ba;
  pos0 = p_gps_m[N_m];
 
  // Propaga R0 fino al timestamp T
  Eigen::Matrix3d R_T = R0;
  for (int i = 0; i < N_m; ++i) R_T = R_T * segs_m[i].delta_R;
  R0 = R_T;

  // Cerca la prima coppia GPS con spostamento significativo
  Eigen::Vector3d heading = Eigen::Vector3d::Zero();
  for (int i = 0; i + 1 < (int)p_gps_m.size(); ++i) {
    Eigen::Vector3d dp = p_gps_m[i+1] - p_gps_m[i];
    dp.z() = 0.0;
    if (dp.norm() > 1.0) {  // spostamento > 2*sigma_GPS
      heading = dp.normalized();
      std::cout << " heading GPS trovato a segmento " << i 
                << "  |dp|=" << dp.norm() << "\n";
      break;
    }
  }
  if (heading.norm() > 0.5) {
      double yaw_gps = std::atan2(heading.y(), heading.x());
      // Decomposizione di R0 in roll/pitch (da gravità) + sostituzione yaw (da GPS)
      Eigen::Vector3d euler = R0.eulerAngles(2, 1, 0);  // ZYX: yaw, pitch, roll
      
      R0 = Eigen::AngleAxisd(yaw_gps,    Eigen::Vector3d::UnitZ()).toRotationMatrix()
          * Eigen::AngleAxisd(euler(1),   Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(euler(2),   Eigen::Vector3d::UnitX()).toRotationMatrix();
      
      std::cout << " yaw da GPS=" << yaw_gps * 180.0/M_PI << "°"
                << "  yaw precedente=" << euler(0) * 180.0/M_PI << "°\n";
  } else {
      std::cout << " heading GPS non affidabile, yaw non corretto\n";
  }
 
  // P0 blocchi posizione e velocità
  Eigen::VectorXd res_m = A_m * x_m - b_m;
  double sigma2_res = res_m.squaredNorm() / std::max(N_m, 1);
  double T_init = t_gps_m.back() - t_gps_m.front();
 
  P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();
  P0.block<3,3>(6,6) = std::max(sigma2_res, 0.01) * Eigen::Matrix3d::Identity();
  P0.block<3,3>(9,9)   = std::max(pow(imu_noise_std(2), 2) * T_init, delta_ba.squaredNorm() / 3.0) * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = pow(imu_noise_std(3), 2) * T_init * Eigen::Matrix3d::Identity();
 
  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [ ";
    for (int col = 0; col < 3; ++col) std::cout << std::setw(9) << R0(row,col) << " ";
    std::cout << "]\n";
  }
  std::cout << " pos0=" << pos0.transpose() << "\n";
  std::cout << " vel0=" << vel0.transpose() << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose() << "\n";
  std::cout << " P0 diag=" << P0.diagonal().transpose() << "\n";
  std::cout << " T_init_moto=" << T_init << " s\n";
  std::cout << "==========  END ==========\n\n";
}

void initialization_5(Eigen::Matrix<double,15,15>& P0,
                      Eigen::Matrix3d&   R0,
                      Eigen::Vector3d&   pos0,
                      Eigen::Vector3d&   vel0,
                      Eigen::Vector3d&   ba0,
                      Eigen::Vector3d&   bg0,
                      ImuInputs&         inputs,
                      std::vector<int>&  imu_indices,
                      GpsMeasurements&   gps_input,
                      const std::vector<double>& dts_imu,
                      double             gps_noise_std,
                      Vector4d           imu_noise_std){
  P0.setZero();
  std::cout << "\n========== [init5] START ==========\n";
  // COLD START
  std::vector<int>             gps_indices;
  std::vector<Eigen::Vector3d> p_gps;
  std::vector<double>          t_gps;
 
  for (int k = 0; k < (int)imu_indices.size() && (int)p_gps.size() < 2; ++k) {
    int imu_idx = imu_indices[k];
    if (imu_idx >= (int)dts_imu.size()) break;
    if (gps_input.count(k)) {
      gps_indices.push_back(imu_idx);
      p_gps.push_back(gps_input[k]);
      t_gps.push_back(dts_imu[imu_idx]);
    }
  }
 
  if ((int)p_gps.size() < 2) {
    std::cerr << " meno di 2 GPS fix disponibili, fallback zero.\n";
    R0 = Eigen::Matrix3d::Identity();
    pos0 = vel0 = ba0 = bg0 = Eigen::Vector3d::Zero();
    P0 = Eigen::Matrix<double,15,15>::Identity() * 1.0;
    return;
  }
 
  const double dt_seg = t_gps[1] - t_gps[0];
  std::cout << " primi due GPS fix: dt=" << dt_seg << "s  campioni IMU nel segmento: " << gps_indices[1] - gps_indices[0] << "\n";

  Eigen::Vector3d acc_sum  = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
  int cnt = 0;
  for (int k = gps_indices[0]; k < gps_indices[1] &&
       k < (int)inputs.size() && k + 1 < (int)dts_imu.size(); ++k) {
    acc_sum  += inputs[k].head<3>();
    gyro_sum += inputs[k].tail<3>();
    ++cnt;
  }
  if (cnt < 2) {
    std::cerr << " troppo pochi campioni IMU nel segmento (" << cnt << ").\n";
    R0 = Eigen::Matrix3d::Identity();
    pos0 = p_gps[1]; vel0 = ba0 = bg0 = Eigen::Vector3d::Zero();
    P0 = Eigen::Matrix<double,15,15>::Identity() * 1.0;
    return;
  }
  Eigen::Vector3d acc_mean  = acc_sum  / cnt;
  Eigen::Vector3d gyro_mean = gyro_sum / cnt;
  std::cout << " campioni IMU usati=" << cnt << "  |acc_mean|=" << acc_mean.norm() << "  a_nongrav=" << std::abs(acc_mean.norm() - 9.81) << "\n";
 
  // Rotazione da allineamento gravità
  Eigen::Vector3d g_meas = acc_mean.normalized();
  Eigen::Vector3d g_world_dir(0.0, 0.0, -1.0);
  Eigen::Vector3d vc = g_meas.cross(g_world_dir);
  double c_dot = g_meas.dot(g_world_dir), s_vc = vc.norm();
  R0 = Eigen::Matrix3d::Identity();
  if (s_vc > 1e-8) {
    Eigen::Matrix3d vx = lie_algebra::skew(vc);
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1.0 - c_dot) / (s_vc * s_vc));
  }
  Eigen::Vector3d g_world_full(0.0, 0.0, -9.81);
  ba0 = acc_mean - R0.transpose() * g_world_full;
  bg0 = gyro_mean;
 
  //  => p1 = p0 + v0*dt + R0*Δp + 0.5*g*dt²
  //  => v0 = (p1 - p0 - R0*Δp - 0.5*g*dt²) / dt
  PreintSegment seg = preintegrate_segment(gps_indices[0], gps_indices[1], inputs, dts_imu, bg0, ba0);
  const Eigen::Vector3d g_known(0.0, 0.0, -9.81);

  Eigen::Vector3d v0_start = (p_gps[1] - p_gps[0] - R0 * seg.delta_p - 0.5 * dt_seg * dt_seg * g_known) / dt_seg;
  vel0 = v0_start + g_known * dt_seg + R0 * seg.delta_v;

  Eigen::Matrix3d R0_end = R0 * seg.delta_R;
  R0 = R0_end;
  
  pos0 = p_gps[1];

  P0(0,0) = 5e-3; P0(1,1) = 5e-3; P0(2,2) = 1.0;
  
  P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();
  
  double sigma2_vel = 2.0 * gps_noise_std * gps_noise_std / (dt_seg * dt_seg);
  P0.block<3,3>(6,6) = sigma2_vel * Eigen::Matrix3d::Identity();
  
  P0.block<3,3>(9,9)   = 10.0 * pow(imu_noise_std(2), 2) * dt_seg * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = 10.0 * pow(imu_noise_std(3), 2) * dt_seg * Eigen::Matrix3d::Identity();
 
  // Output
  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [ ";
    for (int col = 0; col < 3; ++col) std::cout << std::setw(9) << R0(row,col) << " ";
    std::cout << "]\n";
  }
  std::cout << " pos0=" << pos0.transpose() << "\n";
  std::cout << " vel0=" << vel0.transpose()
            << "  |vel0|=" << vel0.norm() << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose() << "\n";
  std::cout << " P0 diag=" << P0.diagonal().transpose() << "\n";
  std::cout << "==========  END ==========\n\n";
}

void initialization_6(Eigen::Matrix<double,15,15>& P0,
                      Eigen::Matrix3d&   R0,
                      Eigen::Vector3d&   pos0,
                      Eigen::Vector3d&   vel0,
                      Eigen::Vector3d&   ba0,
                      Eigen::Vector3d&   bg0,
                      int                STATIC_STEPS,   // campioni statici per R0/bg
                      int                MOTION_STEPS,   // finestra GPS per ba/g
                      ImuInputs&         inputs,
                      std::vector<int>&  imu_indices,
                      GpsMeasurements&   gps_input,
                      const std::vector<double>& dts_imu,
                      double             gps_noise_std,
                      Vector4d           imu_noise_std){
  P0.setZero();
  std::cout << "\n========== [init6] START ==========\n";
  std::cout << "  STATIC_STEPS=" << STATIC_STEPS << "  MOTION_STEPS=" << MOTION_STEPS << "\n";

  // stima di R0 e bias gyro 
  Vector6d mean = Vector6d::Zero();
  for (int i = 0; i < STATIC_STEPS; ++i)
    mean += inputs[imu_indices[i]];
  mean /= STATIC_STEPS;

  const Eigen::Vector3d acc_mean  = mean.head<3>();
  const Eigen::Vector3d gyro_mean = mean.tail<3>();

  Eigen::Vector3d g_meas  = acc_mean.normalized();
  Eigen::Vector3d g_world(0.0, 0.0, -1.0);
  Eigen::Vector3d vc = g_meas.cross(g_world);
  double c = g_meas.dot(g_world), s = vc.norm();
  R0 = Eigen::Matrix3d::Identity();
  if (s > 1e-8) {
    Eigen::Matrix3d vx = lie_algebra::skew(vc);
    R0 = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1.0 - c) / (s * s));
  }
  bg0 = gyro_mean;
  // ba provvisorio per la preintegrazione (verrà sostituito)
  Eigen::Vector3d g_world_full(0.0, 0.0, -9.81);
  ba0 = acc_mean - R0.transpose() * g_world_full;
  
  // stima di g e bias acc
  std::vector<int>             gps_indices;
  std::vector<Eigen::Vector3d> p_gps;
  std::vector<double>          t_gps;

  for (int k = STATIC_STEPS; k < MOTION_STEPS; ++k) {
    if (k >= (int)imu_indices.size()) break;
    int imu_idx = imu_indices[k];
    if (imu_idx >= (int)dts_imu.size()) break;
    if (gps_input.count(k)) {
      p_gps.push_back(gps_input[k]);
      t_gps.push_back(dts_imu[imu_idx]);
      gps_indices.push_back(imu_idx);
    }
  }
  const int N = (int)p_gps.size();  // numero di keyframe GPS
  std::cout << " keyframe GPS in finestra moto: " << N << "\n";
  if (N < 4) {
    std::cerr << " GPS insufficienti (N=" << N << ", serve >=4). Fallback.\n";
    pos0 = vel0 = Eigen::Vector3d::Zero();
    P0.block<3,3>(3,3) = 10.0 * Eigen::Matrix3d::Identity();
    P0.block<3,3>(6,6) = 10.0 * Eigen::Matrix3d::Identity();
    P0(0,0)=1e-3; P0(1,1)=1e-3; P0(2,2)=0.5;
    P0.block<3,3>(9,9)   = 1e-3 * Eigen::Matrix3d::Identity();
    P0.block<3,3>(12,12) = 1e-4 * Eigen::Matrix3d::Identity();
    return;
  }
  // preintegrazione con bias acc provvisorio
  std::vector<PreintSegment> segs;
  segs.reserve(N - 1);
  for (int i = 0; i < N - 1; ++i)
    segs.push_back(preintegrate_segment(gps_indices[i], gps_indices[i+1], inputs, dts_imu, bg0, ba0));

  // ================================================================
  //  Incognite:  x = [Δba(3), g(3)]   dimensione 6
  //
  //  Dalla cinematica per i segmenti i-1→i e i→i+1:
  //    p_i     = p_{i-1} + v_{i-1}*dt_{i-1,i} + R_{i-1}*Δp_{i-1,i} + 0.5*g*dt²_{i-1,i}
  //    p_{i+1} = p_i     + v_i    *dt_{i,i+1}  + R_i    *Δp_{i,i+1}  + 0.5*g*dt²_{i,i+1}
  //
  //    α_k  =  A_k * Δba  +  B_k * g   
  //
  //  dove:
  //    α_k = (p_{k+1}-p_k)/dt_{k,k+1} - (p_k-p_{k-1})/dt_{k-1,k}  [GPS]

  //    A_k = R_{k-1}*J_p_ba_{k-1,k}/dt_{k-1,k}
  //          - R_k  *J_p_ba_{k,k+1}/dt_{k,k+1}
  //          - R_{k-1}*J_v_ba_{k-1,k}       

  //    B_k = -0.5*(dt_{k-1,k} + dt_{k,k+1}) * I

  //    π_k = R_k*Δp_{k,k+1}/dt_{k,k+1}
  //          - R_{k-1}*Δp_{k-1,k}/dt_{k-1,k}
  //          + R_{k-1}*Δv_{k-1,k}               [termine noto]
  //
  //  => riscriviamo come:  α_k - π_k = A_k * Δba + B_k * g
  // ================================================================

  const int NUM_TRIPLES = N - 2;
  Eigen::MatrixXd A_sys = Eigen::MatrixXd::Zero(3 * NUM_TRIPLES, 6);
  Eigen::VectorXd b_sys = Eigen::VectorXd::Zero(3 * NUM_TRIPLES);

  for (int k = 1; k < N - 1; ++k)  // k = indice del keyframe centrale della tripla
  {
    const int row = (k - 1) * 3;
    const double dt_prev = t_gps[k]   - t_gps[k-1];  // dt_{k-1,k}
    const double dt_curr = t_gps[k+1] - t_gps[k];    // dt_{k,k+1}

    // Rotazioni assolute ai keyframe k-1 e k; propaga R0 con le delta_R accumulate
    Eigen::Matrix3d R_prev = R0;
    for (int j = 0; j < k - 1; ++j) R_prev = R_prev * segs[j].delta_R;
    Eigen::Matrix3d R_curr = R_prev * segs[k-1].delta_R;

    // α_k: differenza di "velocità media" GPS tra segmenti adiacenti
    Eigen::Vector3d alpha_k = (p_gps[k+1] - p_gps[k]) / dt_curr - (p_gps[k]   - p_gps[k-1]) / dt_prev;

    // A_k: Jacobiano rispetto a Δba (correzione al ba provvisorio)
    Eigen::Matrix3d A_k = R_prev * segs[k-1].J_p_ba / dt_prev - R_curr * segs[k].J_p_ba   / dt_curr - R_prev * segs[k-1].J_v_ba;

    // B_k: coefficiente di g
    Eigen::Matrix3d B_k = -0.5 * (dt_prev + dt_curr) * Eigen::Matrix3d::Identity();

    // π_k: termine noto dalla preintegrazione nominale
    Eigen::Vector3d pi_k = R_curr * segs[k].delta_p   / dt_curr - R_prev * segs[k-1].delta_p / dt_prev + R_prev * segs[k-1].delta_v;

    // Riempimento del sistema: (α_k - π_k) = A_k*Δba + B_k*g
    A_sys.block<3,3>(row, 0) = A_k;   // Δba
    A_sys.block<3,3>(row, 3) = B_k;   // g
    b_sys.segment<3>(row)    = alpha_k - pi_k;
  }
  Eigen::VectorXd x = A_sys.colPivHouseholderQr().solve(b_sys);
  Eigen::Vector3d delta_ba = x.head<3>();
  Eigen::Vector3d g_est    = x.tail<3>();
  // Normalizza gravità al valore fisico
  std::cout << " g_est (pre-norm): " << g_est.transpose() << "  |g|=" << g_est.norm() << "\n";
  if (g_est.norm() > 1.0) g_est = g_est.normalized() * 9.81;
  else {
    std::cout << " g_est non affidabile, uso fallback\n";
    g_est = Eigen::Vector3d(0.0, 0.0, -9.81);
  }
  // Aggiorna ba0 con la correzione
  ba0 += delta_ba;

  // ripropaga con ba0 definitivo e ottieni Δp, Δv corretti
  std::vector<PreintSegment> segs_final;
  segs_final.reserve(N - 1);
  for (int i = 0; i < N - 1; ++i)
    segs_final.push_back(preintegrate_segment(gps_indices[i], gps_indices[i+1], inputs, dts_imu, bg0, ba0));

  // stima delle velocità
  std::vector<Eigen::Vector3d> velocities(N);
  for (int i = 0; i < N - 1; ++i) {
    const double dt  = t_gps[i+1] - t_gps[i];
    Eigen::Matrix3d Ri = R0;
    for (int j = 0; j < i; ++j) Ri = Ri * segs_final[j].delta_R;
    velocities[i] = (p_gps[i+1] - p_gps[i] - Ri * segs_final[i].delta_p - 0.5 * dt * dt * g_est) / dt;
  }

  // Velocità all'ultimo keyframe (propagata dal penultimo)
  const int last = N - 2;
  Eigen::Matrix3d R_last = R0;
  for (int j = 0; j < last; ++j) R_last = R_last * segs_final[j].delta_R;
  velocities[N-1] = velocities[last] + g_est * (t_gps[N-1] - t_gps[N-2]) + R_last * segs_final[last].delta_v;

  // Valida la velocità: se il robot è ancora lento, non fidarsi
  double v_gps_coarse = (p_gps.back() - p_gps.front()).norm() / (t_gps.back() - t_gps.front());
  std::cout << " v_gps_coarse=" << v_gps_coarse << " m/s\n";
  std::cout << " velocità stimate:\n";
  for (int i = 0; i < N; ++i)
    std::cout << "  v[" << i << "]=" << velocities[i].transpose() << "  |v|=" << velocities[i].norm() << "\n";

  if (v_gps_coarse < 0.3) {
    std::cout << " moto troppo lento per stima vel affidabile, vel0=0\n";
    vel0 = Eigen::Vector3d::Zero();
    P0.block<3,3>(6,6) = 5.0 * Eigen::Matrix3d::Identity();
  } else {
    vel0 = velocities[N-1];
    // Covarianza velocità dai residui del sistema lineare
    Eigen::VectorXd res = A_sys * x - b_sys;
    double sigma2_res = res.squaredNorm() / std::max(3*NUM_TRIPLES - 6, 1);
    P0.block<3,3>(6,6) = std::max(sigma2_res, 0.01) * Eigen::Matrix3d::Identity();
  }

  // Propagazione di R0 fino all'ultimo frame usato per l'inizializzazione
  Eigen::Matrix3d R_T = R0;
  for (int i = 0; i < N - 1; ++i) R_T = R_T * segs_final[i].delta_R;
  R0 = R_T;
  // Posizione = ultimo GPS fix della finestra
  pos0 = p_gps[N-1];

  // definizione della covarianza P0
  P0(0,0) = 1e-3; P0(1,1) = 1e-3; P0(2,2) = 0.5;
  P0.block<3,3>(3,3) = gps_noise_std * gps_noise_std * Eigen::Matrix3d::Identity();

  double T_init = t_gps.back() - t_gps.front();
  double sigma2_ba = std::max(pow(imu_noise_std(2), 2) * T_init, delta_ba.squaredNorm() / 3.0);
  P0.block<3,3>(9,9)   = sigma2_ba * Eigen::Matrix3d::Identity();
  P0.block<3,3>(12,12) = pow(imu_noise_std(1), 2) * T_init * Eigen::Matrix3d::Identity();
  
  std::cout << " R0:\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [ ";
    for (int col = 0; col < 3; ++col)
      std::cout << R0(row,col) << " ";
    std::cout << "]\n";
  }
  Eigen::Vector3d euler = R0.eulerAngles(2,1,0);
  std::cout << " R0 euler: roll=" << euler(2)*180/M_PI << "°  pitch=" << euler(1)*180/M_PI << "°  yaw="   << euler(0)*180/M_PI << "°\n";
  std::cout << " pos0=" << pos0.transpose() << "\n";
  std::cout << " vel0=" << vel0.transpose()  << "  |vel0|=" << vel0.norm() << "\n";
  std::cout << " ba0=" << ba0.transpose() << "\n";
  std::cout << " bg0=" << bg0.transpose() << "\n";
  std::cout << " T_init=" << T_init << " s\n";
  std::cout << " P0 diag=" << P0.diagonal().transpose() << "\n";
  std::cout << "========== [init6] END ==========\n\n";
}

int main(int argc, char** argv){
  if (argc < 2) {
    std::cout << "usage: this-executable path-to-output" << std::endl;
    std::cout << "example: ./example_inizialization /home/giordano/Desktop/traj" << std::endl;
    return 1;
  }

  double gps_noise_std  = 0.005;   // 0.5 
  int gps_freq    = 40;   //  each 40 IMU samples drop a GPS measure (5Hz GPS measures)
  Vector4d imu_noise_std;
  imu_noise_std << 8e-2,   // sigma_noise_accel - 1.86e-03
                   4e-3,   // sigma_noise_gyro  - 1.87e-04
                   4e-5,   // sigma_walk_accel  - 4.33e-04
                   2e-6;   // sigma_walk_gyro   - 2.66e-05
  int INIT_STEPS  = 160;
  // these are the timestamps for which the robot is not moving
  // from timestamp 1413393887225760512 (252 IMU <-> 2  GPS) 
  // to   timestamp 1413393887555760384 (318 IMU <-> 68 GPS)

  ImuStatesExt gt_states, gt_interp;
  ImuInputs inputs;
  std::vector<int> imu_indices;
  GpsMeasurements gps_input;
  IntegrationSteps dts_imu, dts_states; 

  //  V2_01_easy    V2_02_medium    V2_03_difficult
  std::string dataset_filename = data_folder + "/../../vio/datasets/EUROC/V2_01_easy/mav0/imu0/data.csv";
  std::string gt_filename      = data_folder + "/../../vio/datasets/EUROC/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv";

  read_input(dts_imu, inputs, dataset_filename);
  read_states(dts_states, gt_states, gt_filename);

  std::cout << "=== POST READING FUNCTIONS ===\n";
  std::cout << "dts_imu.size()    = " << dts_imu.size()    << "\n";
  std::cout << "inputs.size()     = " << inputs.size()     << "\n";
  std::cout << "gt_states.size()  = " << gt_states.size()  << "\n";
  std::cout << "dts_states.size() = " << dts_states.size() << "\n";
 
  interpolate_groundtruth(dts_imu, dts_states, gt_states, gt_interp, imu_indices);
 
  std::cout << "=== POST INTERPOLATION ===\n";
  std::cout << "gt_interp.size()    = " << gt_interp.size()    << "\n";
  std::cout << "imu_indices.size()  = " << imu_indices.size()  << "\n";
  if (!imu_indices.empty()) {
    std::cout << "imu_indices.front() = " << imu_indices.front() << "\n";
    std::cout << "imu_indices.back()  = " << imu_indices.back()  << "\n";
  }
 
  simulate_gps_data(gps_input, gt_interp, gps_noise_std, gps_freq);
 
  std::cout << "=== POST SIMULATE GPS ===\n";
  std::cout << "gps_input.size() = " << gps_input.size() << "\n";
  if (!gps_input.empty()) {
    std::cout << "gps_input keys: first=" << gps_input.begin()->first
              << "  last=" << gps_input.rbegin()->first << "\n";
  }
  // Verifica quanti GPS cadono nella finestra INIT_STEPS
  int gps_in_window = 0;
  for (auto& kv : gps_input) if (kv.first < INIT_STEPS) gps_in_window++;
  std::cout << "GPS fix in [0, INIT_STEPS=" << INIT_STEPS << "): " << gps_in_window << "\n";
  
  // FILTERS INIZIALIZATION
  NavStateExtended ukf_state_zero_1, ukf_state_zero_2, ukf_state_zero_3, ukf_state_zero_4, ukf_state_zero_5, ukf_state_zero_6, ukf_state_gt_all, ukf_state_zero;

  Matrix12d Q = Matrix12d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
  Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);
  Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  ukf_state_zero.rotation() = Eigen::Matrix3d::Identity();
  ukf_state_zero.velocity() = Eigen::Vector3d(0, 0, 0);
  ukf_state_zero.position() = Eigen::Vector3d(0, 0, 0);
  ukf_state_zero.biasAcc()  = Eigen::Vector3d(0, 0, 0);
  ukf_state_zero.biasGyro() = Eigen::Vector3d(0, 0, 0);
  ukf_state_zero_1  = ukf_state_zero_2 = ukf_state_zero_3 = ukf_state_zero_4 = ukf_state_zero_5 = ukf_state_zero_6 = ukf_state_gt_all = ukf_state_zero;

  int STATIC_STEPS = 160;   // robot fermo: per R0, ba, bg
  int MOTION_STEPS = 800;   // robot in moto: per vel0, pos0

  int start_k_0   = STATIC_STEPS;   // init0: nessuna init, parte da STATIC_STEPS
  int start_k_1   = STATIC_STEPS;   // init1: usa solo campioni statici
  int start_k_2   = STATIC_STEPS;   // init2: usa STATIC_STEPS (GPS nella finestra statica)
  int start_k_3   = MOTION_STEPS;   // init3: stessa finestra di init2
  int start_k_4   = MOTION_STEPS;   // init4: usa MOTION_STEPS (GPS in moto)
  int start_k_5   = 40;             // init5: usa solo i primi 2 GPS fix (~40 campioni)
  int start_k_6   = MOTION_STEPS;   // init6: usa MOTION_STEPS (GPS in moto)
  int start_k_gt  = STATIC_STEPS;              // gt-all: per confronto
  int START_K_METRICS = std::max({start_k_0, start_k_1, start_k_2, start_k_3, start_k_4, start_k_5, start_k_6, start_k_gt});

  Eigen::Matrix<double,15,15> P0;
  Eigen::Matrix3d R0;
  Eigen::Vector3d ba0, bg0, pos0, vel0;
  
  // using first measures of IMU to inizialize with type 1 filter states
  initialization_1(P0, R0, ba0, bg0, STATIC_STEPS, inputs, imu_indices);
  ukf_state_zero_1.rotation()   = R0;
  ukf_state_zero_1.biasAcc()    = ba0;
  ukf_state_zero_1.biasGyro()   = bg0;
  ukf_state_zero_1.covariance() = P0;

  initialization_2(P0, R0, pos0, vel0, ba0, bg0, STATIC_STEPS, inputs, imu_indices, gps_input, dts_imu, gps_noise_std, imu_noise_std);
  ukf_state_zero_2.rotation()   = R0;
  ukf_state_zero_2.position()   = pos0;
  ukf_state_zero_2.velocity()   = vel0;
  ukf_state_zero_2.biasAcc()    = ba0;
  ukf_state_zero_2.biasGyro()   = bg0;
  ukf_state_zero_2.covariance() = P0;

  initialization_3(P0, R0, pos0, vel0, ba0, bg0, STATIC_STEPS, MOTION_STEPS, inputs, imu_indices, gps_input, dts_imu, gps_noise_std, imu_noise_std);
  ukf_state_zero_3.rotation()   = R0;
  ukf_state_zero_3.position()   = pos0;
  ukf_state_zero_3.velocity()   = vel0;
  ukf_state_zero_3.biasAcc()    = ba0;
  ukf_state_zero_3.biasGyro()   = bg0;
  ukf_state_zero_3.covariance() = P0;

  initialization_4(P0, R0, pos0, vel0, ba0, bg0, MOTION_STEPS, inputs, imu_indices, gps_input, dts_imu, gps_noise_std, imu_noise_std);
  ukf_state_zero_4.rotation()   = R0;
  ukf_state_zero_4.position()   = pos0;
  ukf_state_zero_4.velocity()   = vel0;
  ukf_state_zero_4.biasAcc()    = ba0;
  ukf_state_zero_4.biasGyro()   = bg0;
  ukf_state_zero_4.covariance() = P0;

  initialization_5(P0, R0, pos0, vel0, ba0, bg0, inputs, imu_indices, gps_input, dts_imu, gps_noise_std, imu_noise_std);
  ukf_state_zero_5.rotation()   = R0;
  ukf_state_zero_5.position()   = pos0;
  ukf_state_zero_5.velocity()   = vel0;
  ukf_state_zero_5.biasAcc()    = ba0;
  ukf_state_zero_5.biasGyro()   = bg0;
  ukf_state_zero_5.covariance() = P0;

  initialization_6(P0, R0, pos0, vel0, ba0, bg0, STATIC_STEPS, MOTION_STEPS, inputs, imu_indices, gps_input, dts_imu, gps_noise_std, imu_noise_std);
  ukf_state_zero_4.rotation()   = R0;
  ukf_state_zero_4.position()   = pos0;
  ukf_state_zero_4.velocity()   = vel0;
  ukf_state_zero_4.biasAcc()    = ba0;
  ukf_state_zero_4.biasGyro()   = bg0;
  ukf_state_zero_4.covariance() = P0;

  // GT-ALL initialization
  ukf_state_gt_all.position()   = gt_interp[STATIC_STEPS].position();
  ukf_state_gt_all.rotation()   = gt_interp[STATIC_STEPS].rotation();
  ukf_state_gt_all.velocity()   = gt_interp[STATIC_STEPS].velocity();
  ukf_state_gt_all.biasAcc()    = gt_interp[STATIC_STEPS].biasAcc();
  ukf_state_gt_all.biasGyro()   = gt_interp[STATIC_STEPS].biasGyro();
  ukf_state_gt_all.covariance() = gt_interp[STATIC_STEPS].covariance();

  std::cout << "[GT(0)] R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << gt_interp[0].rotation()(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << "[GT(0)] pos0="  << gt_interp[0].position().transpose()  << "\n";
  std::cout << "[GT(0)] vel0="  << gt_interp[0].velocity().transpose()  << "\n";
  std::cout << "[GT(0)] ba0=" << gt_interp[0].biasAcc().transpose() << "\n";
  std::cout << "[GT(0)] bg0=" << gt_interp[0].biasGyro().transpose()   << "\n";
  std::cout << "[GT(0)] P0 diag=" << gt_interp[0].covariance().diagonal().transpose() << "\n";

  std::cout << "[GT(STATIC_STEPS)] R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << gt_interp[STATIC_STEPS].rotation()(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << "[GT(STATIC_STEPS)] pos0="  << gt_interp[STATIC_STEPS].position().transpose()  << "\n";
  std::cout << "[GT(STATIC_STEPS)] vel0="  << gt_interp[STATIC_STEPS].velocity().transpose()  << "\n";
  std::cout << "[GT(STATIC_STEPS)] ba0=" << gt_interp[STATIC_STEPS].biasAcc().transpose() << "\n";
  std::cout << "[GT(STATIC_STEPS)] bg0=" << gt_interp[STATIC_STEPS].biasGyro().transpose()   << "\n";
  std::cout << "[GT(STATIC_STEPS)] P0 diag=" << gt_interp[STATIC_STEPS].covariance().diagonal().transpose() << "\n";

  std::cout << "[GT(0)] R0:\n";
  for (int row = 0; row < 3; ++row) {
      std::cout << "  [ ";
      for (int col = 0; col < 3; ++col)
          std::cout << gt_interp[0].rotation()(row, col) << " ";
      std::cout << "]\n";
  }
  std::cout << "[GT(MOTION_STEPS)] pos0="  << gt_interp[MOTION_STEPS].position().transpose()  << "\n";
  std::cout << "[GT(MOTION_STEPS)] vel0="  << gt_interp[MOTION_STEPS].velocity().transpose()  << "\n";
  std::cout << "[GT(MOTION_STEPS)] ba0=" << gt_interp[MOTION_STEPS].biasAcc().transpose() << "\n";
  std::cout << "[GT(MOTION_STEPS)] bg0=" << gt_interp[MOTION_STEPS].biasGyro().transpose()   << "\n";
  std::cout << "[GT(MOTION_STEPS)] P0 diag=" << gt_interp[MOTION_STEPS].covariance().diagonal().transpose() << "\n";

  // Trova il primo k dove il robot inizia a muoversi
  for (int k = 0; k < std::min(3000, (int)gt_interp.size()); k += 40) {
    double v = gt_interp[k].velocity().norm();
    if (v > 0.1){
      std::cout << "Moto rilevato nel GT a k=" << k << "  |vel|=" << v << "\n";
      break;
    }
    std::cout << "k=" << k << "  |vel_GT|=" << v << "\n";
  }

  // ------- UKF filter
  UKFImuExtendedGps ukf_0, ukf_01, ukf_02, ukf_03, ukf_04, ukf_05, ukf_06, ukf_gtAll;
  ukf_0.setWeights(ukf_state_zero.stateDim(), 6, alpha);
  ukf_01.setWeights(ukf_state_zero_1.stateDim(), 6, alpha);
  ukf_02.setWeights(ukf_state_zero_2.stateDim(), 6, alpha);
  ukf_03.setWeights(ukf_state_zero_3.stateDim(), 6, alpha);
  ukf_04.setWeights(ukf_state_zero_4.stateDim(), 6, alpha);
  ukf_05.setWeights(ukf_state_zero_5.stateDim(), 6, alpha);
  ukf_06.setWeights(ukf_state_zero_6.stateDim(), 6, alpha);
  ukf_gtAll.setWeights(ukf_state_gt_all.stateDim(), 6, alpha);

  // -------- initializing estimates
  int N = imu_indices.size();  

  std::vector<NavStateExtended> estimates_0(N), estimates_01(N), estimates_02(N), estimates_03(N), 
                                estimates_04(N), estimates_05(N), estimates_06(N), estimates_gtAll(N);
  estimates_0.at(start_k_0)      = ukf_state_zero;
  estimates_01.at(start_k_1)     = ukf_state_zero_1;
  estimates_02.at(start_k_2)     = ukf_state_zero_2;
  estimates_03.at(start_k_3)     = ukf_state_zero_3;
  estimates_04.at(start_k_4)     = ukf_state_zero_4;
  estimates_05.at(start_k_5)     = ukf_state_zero_5;
  estimates_06.at(start_k_6)     = ukf_state_zero_6;
  estimates_gtAll.at(start_k_gt) = ukf_state_gt_all;

  GPSObs obs;
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * pow(gps_noise_std, 2);
  double DT;
  const Matrix6d Q_bias = Q.block<6,6>(6,6);
  const Matrix6d Q_inputs = Q.block<6, 6>(0, 0);

  std::cout << "=== STARTING FILTER LOOP ===\n";

  // START FILTERING MEASURES
  for (int k = 1; k < N; ++k) {
    if (k%1000 == 0) std::cout << "--- iteration: " << k-INIT_STEPS << " ---\n";
    
    int i = imu_indices[k];
    int i_prev = imu_indices[k-1];
    
    DT = dts_imu[i] - dts_imu[i_prev];
    if (DT <= 0) continue;

    if (k > start_k_0) {
      ukf_0.propagate(ukf_state_zero, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k)) {
        obs.meas() = gps_input.at(k);
        ukf_0.update(ukf_state_zero, obs, R);
      }
      estimates_0.at(k) = ukf_state_zero;
    }
    if (k > start_k_1) {
      ukf_01.propagate(ukf_state_zero_1, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_1.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_01.update(ukf_state_zero_1, obs, R);
      }
      estimates_01.at(k) = ukf_state_zero_1;
    }
    if (k > start_k_2) {
      ukf_02.propagate(ukf_state_zero_2, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_2.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_02.update(ukf_state_zero_2, obs, R);
      }
      estimates_02.at(k) = ukf_state_zero_2;
    }
    if (k > start_k_3) {
      ukf_03.propagate(ukf_state_zero_3, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_3.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_03.update(ukf_state_zero_3, obs, R);
      }
      estimates_03.at(k) = ukf_state_zero_3;
    }
    if (k > start_k_4) {
      ukf_04.propagate(ukf_state_zero_4, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_4.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_04.update(ukf_state_zero_4, obs, R);
      }
      estimates_04.at(k) = ukf_state_zero_4;
    }
    if (k > start_k_5) {
      ukf_05.propagate(ukf_state_zero_5, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_5.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_05.update(ukf_state_zero_5, obs, R);
      }
      estimates_05.at(k) = ukf_state_zero_5;
    }
    if (k > start_k_6) {
      ukf_06.propagate(ukf_state_zero_6, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_zero_6.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_06.update(ukf_state_zero_6, obs, R);
      }
      estimates_06.at(k) = ukf_state_zero_6;
    }
    if (k > start_k_gt) {
      ukf_gtAll.propagate(ukf_state_gt_all, inputs.at(i_prev), Q_inputs, DT);
      ukf_state_gt_all.covariance().block<6,6>(9,9) += Q_bias * DT * DT;
      if (gps_input.count(k))  {
        obs.meas() = gps_input.at(k);
        ukf_gtAll.update(ukf_state_gt_all, obs, R);
      }
      estimates_gtAll.at(k) = ukf_state_gt_all;
    }
  }
  std::cout << "=== ENDING FILTER LOOP ===\n";

  int size = estimates_0.size();
  for (int i = START_K_METRICS; i < size; i++) {
    if (!estimates_0[i].position().allFinite())      { std::cout << "NaN estimates_0 at i="      << i << "\n"; break; }
    if (!estimates_01[i].position().allFinite())     { std::cout << "NaN estimates_01 at i="     << i << "\n"; break; }
    if (!estimates_02[i].position().allFinite())     { std::cout << "NaN estimates_02 at i="     << i << "\n"; break; }
    if (!estimates_03[i].position().allFinite())     { std::cout << "NaN estimates_03 at i="     << i << "\n"; break; }
    if (!estimates_04[i].position().allFinite())     { std::cout << "NaN estimates_04 at i="     << i << "\n"; break; }
    if (!estimates_05[i].position().allFinite())     { std::cout << "NaN estimates_05 at i="     << i << "\n"; break; }
    if (!estimates_06[i].position().allFinite())     { std::cout << "NaN estimates_06 at i="     << i << "\n"; break; }
    if (!estimates_gtAll[i].position().allFinite())  { std::cout << "NaN estimates_gtAll at i="  << i << "\n"; break; }
    if (!gt_interp[i].position().allFinite())        { std::cout << "NaN gt_interp at i="        << i << "\n"; break; }
  }
  
  ImuStatesExt est0_valid(  estimates_0.begin()    + START_K_METRICS, estimates_0.end());
  ImuStatesExt est01_valid( estimates_01.begin()   + START_K_METRICS, estimates_01.end());
  ImuStatesExt est02_valid( estimates_02.begin()   + START_K_METRICS, estimates_02.end());
  ImuStatesExt est03_valid( estimates_03.begin()   + START_K_METRICS, estimates_03.end());
  ImuStatesExt est04_valid( estimates_04.begin()   + START_K_METRICS, estimates_04.end());
  ImuStatesExt est05_valid( estimates_05.begin()   + START_K_METRICS, estimates_05.end());
  ImuStatesExt est06_valid( estimates_06.begin()   + START_K_METRICS, estimates_06.end());
  ImuStatesExt estgtA_valid(estimates_gtAll.begin()+ START_K_METRICS, estimates_gtAll.end());
  ImuStatesExt gt_valid(    gt_interp.begin()      + START_K_METRICS, gt_interp.end());
 
  std::cout << "\n       UKF 0     :\n";
  std::cout << " ATE             : " << computeATE2ext(est0_valid,  gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est0_valid,  gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est0_valid,  gt_valid) << "\n";
  std::cout << "       UKF 01    :\n";
  std::cout << " ATE             : " << computeATE2ext(est01_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est01_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est01_valid, gt_valid) << "\n";
  std::cout << "       UKF 02    :\n";
  std::cout << " ATE             : " << computeATE2ext(est02_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est02_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est02_valid, gt_valid) << "\n";
  std::cout << "       UKF 03    :\n";
  std::cout << " ATE             : " << computeATE2ext(est03_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est03_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est03_valid, gt_valid) << "\n";
  std::cout << "       UKF 04    :\n";
  std::cout << " ATE             : " << computeATE2ext(est04_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est04_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est04_valid, gt_valid) << "\n";
  std::cout << "       UKF 05    :\n";
  std::cout << " ATE             : " << computeATE2ext(est05_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est05_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est05_valid, gt_valid) << "\n";
  std::cout << "       UKF 06    :\n";
  std::cout << " ATE             : " << computeATE2ext(est06_valid, gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(est06_valid, gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(est06_valid, gt_valid) << "\n";
  std::cout << "    UKF GT-ALL   :\n";
  std::cout << " ATE             : " << computeATE2ext(estgtA_valid,gt_valid) << "\n";
  std::cout << " RPE translation : " << computeRPETExt(estgtA_valid,gt_valid) << "\n";
  std::cout << " RPE rotation    : " << computeRPERExt(estgtA_valid,gt_valid) << "\n";
  
    // output file
  auto write_traj = [&](const std::string& fname, const ImuStatesExt& states){
    std::ofstream f(fname);
    if (f.is_open())
      for (const auto& s : states)
        f << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
  std::cout << " wrote file    : " << fname << " / len file    : " << states.size() << "\n";
  };

  std::string out = std::string(argv[1]);
  write_traj(out+"/gt_states.txt",    gt_valid);
  write_traj(out+"/estimated_0.txt",  est0_valid);
  write_traj(out+"/estimated_01.txt",  est01_valid);
  write_traj(out+"/estimated_02.txt",  est02_valid);
  write_traj(out+"/estimated_03.txt",  est03_valid);
  write_traj(out+"/estimated_04.txt",  est04_valid);
  write_traj(out+"/estimated_05.txt",  est05_valid);
  write_traj(out+"/estimated_06.txt",  est06_valid);
  write_traj(out+"/estimated_gtAll.txt",  estgtA_valid);

  std::cout << "job finished." << std::endl;

  return 0;
}
