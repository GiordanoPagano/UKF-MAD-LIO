#pragma once
#include "lie_algebra.h"
#include <iostream>
#include <memory>

namespace ukf_manifold {

  using Vector9d = Eigen::Matrix<double, 9, 1>;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;

  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  class NavState {
  public:
    NavState();
    virtual ~NavState() = default;

    void transition(const Vector6d& input, const Vector6d& noise, const double& dt);
    std::unique_ptr<NavState> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const NavState& state_hat);

    // accessors
    inline const Eigen::Matrix3d& rotation() const {
      return R_;
    }
    inline const Eigen::Vector3d& position() const {
      return p_;
    }
    inline const Eigen::Vector3d& velocity() const {
      return v_;
    }
    inline const Matrix9d& covariance() const {
      return covariance_;
    }
    inline const Eigen::Vector3d& gravity() const {
      return g_;
    }

    // setters
    inline Eigen::Matrix3d& rotation() {
      return R_;
    }
    inline Eigen::Vector3d& position() {
      return p_;
    }
    inline Eigen::Vector3d& velocity() {
      return v_;
    }
    inline Matrix9d& covariance() {
      return covariance_;
    }
    inline Eigen::Vector3d& gravity() {
      return g_;
    }

    // dim state for dynamic shit
    inline const int stateDim() const {
      return state_dim_;
    }
    inline const int inputDim() const {
      return input_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const NavState& state);

  protected:
    Eigen::Matrix3d R_   = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_   = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_   = Eigen::Vector3d::Zero();
    Matrix9d covariance_ = Matrix9d::Zero();
    Eigen::Vector3d g_   = Eigen::Vector3d(0.d, 0.d, -9.81);
    size_t state_dim_    = Matrix9d::RowsAtCompileTime;
    size_t input_dim_    = 6;
  };

  using Vector15d = Eigen::Matrix<double, 15, 1>;
  using Matrix15d = Eigen::Matrix<double, 15, 15>;

  class NavStateExtended : public NavState {
  public:
    // static constexpr int StateDim = 15;
    // static constexpr int InputDim = 6;

    NavStateExtended();
    virtual ~NavStateExtended() = default;

    void transition(const Vector6d& input, const Vector6d& noise, const double& dt);
    std::unique_ptr<NavStateExtended> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const NavStateExtended& state_hat);

    inline const Eigen::Vector3d& biasAcc() const {
      return bias_acc_;
    } // getters
    inline const Eigen::Vector3d& biasGyro() const {
      return bias_gyro_;
    }
    inline const Matrix15d& covariance() const {
      return covariance_;
    }

    inline Eigen::Vector3d& biasAcc() {
      return bias_acc_;
    } // setters
    inline Eigen::Vector3d& biasGyro() {
      return bias_gyro_;
    }
    inline Matrix15d& covariance() {
      return covariance_;
    }

    inline const int stateDim() const {
      return state_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const NavStateExtended& state);

  protected:
    Eigen::Vector3d bias_acc_  = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_gyro_ = Eigen::Vector3d::Zero();
    Matrix15d covariance_      = Matrix15d::Zero();
    int state_dim_             = Matrix15d::RowsAtCompileTime;
  };

  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Vector24d = Eigen::Matrix<double, 24, 1>;
  using Matrix24d = Eigen::Matrix<double, 24, 24>;

  class NavStateLidar : public NavStateExtended {
  public:
    // state dimension: 24
    // x = [R_wi, p_wi, v_wi, b_gyr, b_acc, g, R_li, t_li]

    NavStateLidar();
    virtual ~NavStateLidar() = default;

    void transition(
      const Eigen::Vector3d& acc_input, 
      const Eigen::Vector3d& gyro_input, 
      const Eigen::Vector3d& acc_noise, 
      const Eigen::Vector3d& gyro_noise, 
      const Eigen::Vector3d& acc_bias_noise, 
      const Eigen::Vector3d& gyro_bias_noise, 
      const double&   dt);
    std::unique_ptr<NavStateLidar> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const NavStateLidar& state_hat);

    // GETTERS
    inline const Eigen::Matrix3d& R_li() const {
      return R_li_;
    }
    inline const Eigen::Vector3d& t_li() const {
      return t_li_;
    }
    inline const Matrix24d& covariance() const {
      return covariance_;
    }
    inline const bool& using_extr() const {
      return Extr_Estim_;
    }

    // SETTERS
    inline Matrix24d& covariance() {
      return covariance_;
    }
    inline Eigen::Matrix3d& R_li() {
      return R_li_;
    }
    inline Eigen::Vector3d& t_li() {
      return t_li_;
    }

    inline const int stateDim() const {
      return state_dim_;
    }
    inline const int inputDim() const {
      return input_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const NavStateLidar& state);

  protected:
    Eigen::Matrix3d R_li_      = Eigen::Matrix3d::Zero();
    Eigen::Vector3d t_li_      = Eigen::Vector3d::Zero();
    bool Extr_Estim_           = false;
    Matrix24d covariance_      = Matrix24d::Zero();
    int state_dim_             = Matrix24d::RowsAtCompileTime;
    size_t input_dim_          = 12;  // used for UKF prediction
  };

} // namespace ukf_manifold
