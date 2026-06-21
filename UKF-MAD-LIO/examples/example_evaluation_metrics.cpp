#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// -------------------------------------------- Utilities for usage: -----------------------------------------
//
// ./example_evaluation_metrics /home/giordano/Desktop/traj 1 amplitude
//
// ./this-executable path-to-output trajectory-type variation-type
//
// ------------------------------------------- Utilities for plotting: ----------------------------------------
//
//  plot "ATE-RPE_results_NavExt.txt" using 2 w l, "ATE-RPE_results_NavExt.txt" using 3 w l, "ATE-RPE_results_NavExt.txt" using 4 w l, 
//       "ATE-RPE_results_NavExt.txt" using 5 w l, "ATE-RPE_results_Nav.txt" using 2 w l, "ATE-RPE_results_Nav.txt" using 3 w l, 
//       "ATE-RPE_results_Nav.txt" using 4 w l, "ATE-RPE_results_Nav.txt" using 5 w l
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ukf_manifold;

const std::string data_folder(UKF_DATA_FOLDER);

using ImuStates    = std::vector<NavState>;
using ImuStatesExt = std::vector<NavStateExtended>;
using ImuInputs    = std::vector<Vector6d>;
using GpsMeasurements = std::map<int, Eigen::Vector3d>;

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
double computeATE(const ImuStates& estimates, const ImuStates& gt_){
    double error = 0;
    for (size_t i = 0; i < gt_.size(); ++i){
      error += (gt_[i].position() - estimates[i].position()).squaredNorm();
    }
    return std::sqrt(error / gt_.size());
}
double computeATE2(const ImuStates& est, const ImuStates& gt){
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


void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, ImuInputs& inputs_bias_, ImuInputs& inputs_dirt_, 
                       const Vector4d& imu_noise_std_, const std::string& traj_name_,
                       const int T, const double radius, const int imu_freq, const int N, const double dt);

void simulate_gps_data(GpsMeasurements& zs_gps, const ImuStates& states, const double& gps_noise_std, const int gps_freq);


int main(int argc, char** argv){
    if (argc < 4) {
        std::cout << "usage: this-executable path-to-output trajectory-type variation-type" << std::endl;
        std::cout << "example: ./example_evaluation_metrics /home/giordano/Desktop/traj 1 amplitude" << std::endl;
        return 1;
    }
    
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

    std::string variation;
    if (argc >= 4) {
      try {
        variation = std::string(argv[3]);
      }
      catch (const std::exception&) {
        std::cerr << "variation-type must be 'biases' or 'amplitude'\n" << std::endl;
        return 1;
      }
    }
    if (variation != "biases" && variation != "amplitude"){
        std::cerr << "you interted: " << variation << std::endl;
        std::cerr << "Invalid variation-type.\n"
                  << "---> biases\n"
                  << "---> amplitude\n";
        return 1;
    }
    

    Eigen::ArrayXd bias_multipliers = Eigen::ArrayXd::LinSpaced(101, 0, 100);
    Eigen::ArrayXd amplitudes       = Eigen::ArrayXd::LinSpaced(101, 0, 100);
    amplitudes(0) = 0.5;
    const Eigen::ArrayXd& sweep = (variation == "biases") ? bias_multipliers : amplitudes;

    double gps_noise_std  = 0.5;
    Vector4d imu_noise_std;
    imu_noise_std << 1.86e-03, 1.87e-04, 4.33e-04, 2.66e-05;

    int T           = 10;
    double radius   = 10;
    int imu_freq    = 100;
    int gps_freq    = 5;
    int N           = T * imu_freq;
    double dt       = 1.0 / imu_freq;

    std::ofstream results(std::string(argv[1]) + "/ATE-RPE_results_Nav.txt");
    std::ofstream results_ext(std::string(argv[1]) + "/ATE-RPE_results_NavExt.txt");
    std::ofstream results_cov(std::string(argv[1]) + "/ATE_cov_NavExt.txt");
    results << " ATE --- ATE2 --- RPE_T --- RPE_R" << std::endl;
    results_ext << " ATE --- ATE2 --- RPE_T --- RPE_R" << std::endl;
    results_cov << "Covariances multiplier:   0   dt^2   dt   1   1/dt   1/dt^2" << std::endl;
    
    for(auto value : sweep){
        if (variation == "biases") std::cout << "Running with bias multiplier: " << value << std::endl;
        else std::cout << "Running with amplitude: " << value << std::endl;

        ImuStates gt_states;
        ImuStatesExt init_state_ext;
        ImuInputs inputs, inputs_bias, inputs_dirt;
        GpsMeasurements zs_gps;
        
        imu_noise_std << 1.86e-03, 1.87e-04, 4.33e-04, 2.66e-05;
        if (variation == "biases"){
            imu_noise_std(0) *= value;
            imu_noise_std(1) *= value;
            imu_noise_std(2) *= value;
            imu_noise_std(3) *= value;
        }
        else if (variation == "amplitude"){
            radius = value;
            imu_noise_std(0) *= 10;
            imu_noise_std(1) *= 10;
            imu_noise_std(2) *= 10;
            imu_noise_std(3) *= 10;
        }

        simulate_imu_data(gt_states, inputs, inputs_bias, inputs_dirt, imu_noise_std, traj_name, T, radius, imu_freq, N, dt);

        simulate_gps_data(zs_gps, gt_states, gps_noise_std, gps_freq);

        // inizialization of filters
        Matrix12d Q = Matrix12d::Identity();
        Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
        Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
        Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);
        Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);
        const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

        NavStateExtended ukf_ext_state_dirt, ukf_ext_state_bias, ukf_ext_1, ukf_ext_2, ukf_ext_3, ukf_ext_4, ukf_ext_5;
        ukf_ext_state_dirt.rotation() = gt_states[0].rotation();
        ukf_ext_state_dirt.velocity() = gt_states[0].velocity();
        ukf_ext_state_dirt.position() = gt_states[0].position();
        ukf_ext_state_dirt.biasAcc()  = Eigen::Vector3d(0, 0, 0);
        ukf_ext_state_dirt.biasGyro() = Eigen::Vector3d(0, 0, 0);
        ukf_ext_state_bias  = ukf_ext_state_dirt;
        ukf_ext_1 = ukf_ext_2 = ukf_ext_3 = ukf_ext_4 = ukf_ext_5 = ukf_ext_state_dirt;
        NavState         ukf_state           = gt_states.at(0);
        NavState         ukf_state_bias      = gt_states.at(0);
        NavState         ukf_state_dirt      = gt_states.at(0);


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
        // copy all of state_dirt ---> 1_2_3_4_5
        ukf_ext_1 = ukf_ext_2 = ukf_ext_3 = ukf_ext_4 = ukf_ext_5 = ukf_ext_state_dirt;

        // ------- UKF filter
        UKFImuExtendedGps ukf_ext_dirt, ukf_ext_bias, ukf1, ukf2, ukf3, ukf4, ukf5;
        ukf_ext_dirt.setWeights(ukf_ext_state_dirt.stateDim(), 6, alpha);
        ukf_ext_bias.setWeights(ukf_ext_state_bias.stateDim(), 6, alpha);
        UKFImuGps ukf, ukf_bias, ukf_dirt;
        ukf.setWeights(ukf_state.stateDim(), 6, alpha);
        ukf_bias.setWeights(ukf_state_bias.stateDim(), 6, alpha);
        ukf_dirt.setWeights(ukf_state_dirt.stateDim(), 6, alpha);

        ukf1.setWeights(ukf_ext_1.stateDim(), 6, alpha);
        ukf2.setWeights(ukf_ext_1.stateDim(), 6, alpha);
        ukf3.setWeights(ukf_ext_1.stateDim(), 6, alpha);
        ukf4.setWeights(ukf_ext_1.stateDim(), 6, alpha);
        ukf5.setWeights(ukf_ext_1.stateDim(), 6, alpha);

        // ------- start from init gt state
        std::vector<NavStateExtended> estimates_ext_dirt(N), estimates_ext_bias(N), estim1(N), estim2(N), estim3(N), estim4(N), estim5(N);
        estimates_ext_dirt.at(0) = ukf_ext_state_dirt;
        estimates_ext_bias.at(0) = ukf_ext_state_bias;
        std::vector<NavState> estimates(N), estimates_bias(N), estimates_dirt(N);
        estimates.at(0) = ukf_state;
        estimates_bias.at(0) = ukf_state_bias;
        estimates_dirt.at(0) = ukf_state_dirt;
        estim1.at(0) = estim2.at(0) = estim3.at(0) = estim4.at(0) = estim5.at(0) = ukf_ext_state_dirt;

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

            ukf1.propagate(ukf_ext_1, inputs_dirt.at(i - 1), Q_inputs, dt);
            ukf2.propagate(ukf_ext_2, inputs_dirt.at(i - 1), Q_inputs, dt);
            ukf3.propagate(ukf_ext_3, inputs_dirt.at(i - 1), Q_inputs, dt);
            ukf4.propagate(ukf_ext_4, inputs_dirt.at(i - 1), Q_inputs, dt);
            ukf5.propagate(ukf_ext_5, inputs_dirt.at(i - 1), Q_inputs, dt);

            // add bias covariance only to extended state
            // questa cosa in teoria viene fatta per evitare che l'UKF si fossilizzi su un valore di bias 
            // senza aggiornarlo in modo corretto negli step successivi; però nel calcolo del valore ATE
            // sta cosa non porta a miglioramenti effettivi: l'errore rimane costante sia che c'è sia che non c'è
            // forse sarebbe utile nel caso in cui il bias inizia a variare velocemente...
            const Matrix6d Q_bias = Q.block<6, 6>(6, 6);
            ukf_ext_state_dirt.covariance().block<6, 6>(9, 9) += Q_bias * dt * dt;  // adding a very small quantity proportional to Q_bias
            ukf_ext_state_bias.covariance().block<6, 6>(9, 9) += Q_bias * dt * dt;

            ukf_ext_1.covariance().block<6, 6>(9, 9) += Q_bias * 0;        // NO covariance adjust
            ukf_ext_2.covariance().block<6, 6>(9, 9) += Q_bias * dt;       // adding a small quantity proportional to Q_bias
            ukf_ext_3.covariance().block<6, 6>(9, 9) += Q_bias * 1;        // adding Q_bias
            ukf_ext_4.covariance().block<6, 6>(9, 9) += Q_bias / dt;       // adding a big quantity proportional to Q_bias
            ukf_ext_5.covariance().block<6, 6>(9, 9) += Q_bias / dt / dt;  // adding a very big quantity proportional to Q_bias

            if (zs_gps.count(i) > 0) {
                // here update
                const Eigen::Vector3d& curr_meas = zs_gps.at(i);
                obs.meas() = curr_meas;
                ukf_ext_dirt.update(ukf_ext_state_dirt, obs, R);
                ukf_ext_bias.update(ukf_ext_state_bias, obs, R);
                ukf.update(ukf_state, obs, R);
                ukf_bias.update(ukf_state_bias, obs, R);
                ukf_dirt.update(ukf_state_dirt, obs, R);

                ukf1.update(ukf_ext_1, obs, R);
                ukf2.update(ukf_ext_2, obs, R);
                ukf3.update(ukf_ext_3, obs, R);
                ukf4.update(ukf_ext_4, obs, R);
                ukf5.update(ukf_ext_5, obs, R);
            }
            estimates_ext_dirt.at(i) = ukf_ext_state_dirt;
            estimates_ext_bias.at(i) = ukf_ext_state_bias;
            estimates.at(i) = ukf_state;
            estimates_bias.at(i) = ukf_state_bias;
            estimates_dirt.at(i) = ukf_state_dirt;

            estim1.at(i) = ukf_ext_1;
            estim2.at(i) = ukf_ext_2;
            estim3.at(i) = ukf_ext_3;
            estim4.at(i) = ukf_ext_4;
            estim5.at(i) = ukf_ext_5;
        }

        results << value << " " << computeATE(estimates_dirt, gt_states) 
                                    << " " << computeATE2(estimates_dirt, gt_states) 
                                    << " " << computeRPET(estimates_dirt, gt_states) 
                                    << " " << computeRPER(estimates_dirt, gt_states) << std::endl;

        results_ext << value << " " << computeATEext(estimates_ext_dirt, gt_states) 
                                    << " " << computeATE2ext(estimates_ext_dirt, gt_states) 
                                    << " " << computeRPETExt(estimates_ext_dirt, gt_states) 
                                    << " " << computeRPERExt(estimates_ext_dirt, gt_states) << std::endl;

        results_cov << value << " " << computeATE2ext(estim1, gt_states)                       // 0
                                    << " " << computeATE2ext(estimates_ext_dirt, gt_states)    // dt^2
                                    << " " << computeATE2ext(estim2, gt_states)                // dt
                                    << " " << computeATE2ext(estim3, gt_states)                // 1
                                    << " " << computeATE2ext(estim4, gt_states)                // 1/dt
                                    << " " << computeATE2ext(estim5, gt_states)  << std::endl; // 1/dt^2
    }

    results.close();
    results_ext.close();
    results_cov.close();

    std::cout << "job finished." << std::endl;
    //std::cout << "biases: " << imu_noise_std << std::endl;

    return 0;
}


void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, ImuInputs& inputs_bias_, ImuInputs& inputs_dirt_, 
                       const Vector4d& imu_noise_std_, const std::string& traj_name_,
                       const int T, const double radius, const int imu_freq, const int N, const double dt) {
                        
  Eigen::Vector3d g = Eigen::Vector3d(0.0, 0.0, -9.81);

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
