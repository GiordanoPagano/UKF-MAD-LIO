#pragma once
#include "nav_state.h"
#include "observations.h"
#include <Eigen/Cholesky>
#include <memory>
#include <fstream>
#include <iomanip>

namespace ukf_manifold {

  struct Weight {
    double lambda_      = 0.0;
    double sqrt_lambda_ = 0.0;
    double wj_          = 0.0;
    double wm0_         = 0.0;
    double wc0_         = 0.0;
  };

  struct Weights {
    Weight w_d;
    Weight w_q;
    Weight w_u;
  };

  template <class StateType, class ObservationType>
  class UKF_ {
  public:

    UKF_() {
      /*
      f_log.open("/home/giordano/Desktop/UKF_ICP/error_gain-error.txt");
      if (!f_log.is_open()) std::cerr << "Error opening log file!" << std::endl;
      f_log << std::fixed << std::setprecision(9);
      f_log << "timestamp error gain_error   z_bar_x   z_bar_y   z_bar_z\n";

      f_pipe.open("/home/giordano/Desktop/UKF_ICP/pipeline_pos.txt");
      if (!f_pipe.is_open()) std::cerr << "Error opening log file!" << std::endl;
      f_pipe << std::fixed << std::setprecision(9);
      //f_pipe << "timestamp   pose_x   pose_y   pose_z\n";
      */
    }

    ~UKF_() {
      if (f_log.is_open()) f_log.close();
      if (f_pipe.is_open()) f_pipe.close();
    }

    void propagate(StateType& state, const Eigen::VectorXd& input, const Eigen::MatrixXd& CholQ, const double& dt = 0.0);
    Eigen::VectorXd update(StateType& state, ObservationType& z, const Eigen::MatrixXd& R);

    inline void setWeights(const Weights& ws) {
      ws_ = ws;
    }

    inline void setWeights(const int& d, const int& q, const Eigen::Vector3d& alpha) {
      // params for state propagation wrt uncertainty
      ws_.w_d = setWeight(d, alpha(0));
      // params for state propagation wrt noise
      ws_.w_q = setWeight(q, alpha(1));
      // params for state update
      ws_.w_u = setWeight(d, alpha(2));
    }

    inline const Weights& weights() const {
      return ws_;
    }

    template<typename T>
    void write_log(const T& value) {
        f_log << value;
    }

    template<typename T>
    void write_pipe(const T& value) {
        f_pipe << value;
    }
    

  protected:
    Weight setWeight(const int n, const double alpha);
    Weights ws_;
    const double tol_ = 1e-9;
    std::ofstream f_log, f_pipe;
  };

  using UKFImuGps          = UKF_<NavState, GPSObs>;
  using UKFImuExtendedGps  = UKF_<NavStateExtended, GPSObs>;
  using UKFImuExtendedOdom = UKF_<NavStateExtended, OdomObs>;
  using UKFImuLidar        = UKF_<NavStateLidar, OdomObs>;

} // namespace ukf_manifold

#include "ukf.hpp"
