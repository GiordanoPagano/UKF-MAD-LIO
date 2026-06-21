#pragma once
#include "nav_state.h"
#include <Eigen/Dense>

namespace ukf_manifold {
  // gps observation model
  class GPSObs {
  public:
    template <typename StateType>
    Eigen::Vector3d const observation(const StateType& state) const {
      return state.position();
    }
    
    Eigen::Vector3d const obs_dist(const Eigen::Vector3d& zi) const {
      Eigen::Vector3d error;
      error = measure_ - zi;
      return error;
    }

    //template <typename ObservationType>
    GPSObs const compute_mean_obs(const std::vector<GPSObs>& obs_vec){

      GPSObs z_bar;
      const int N = obs_vec.size();
      Eigen::Vector3d t_mean = Eigen::Vector3d::Zero();

      // Computing MEAN POSITION 
      for(const auto& obs : obs_vec) t_mean += obs.meas();
      t_mean /= static_cast<double>(N);

      z_bar.meas() = t_mean;

      return z_bar;
    }

    inline const size_t obsDim() const {
      return 3;
    }

    // getter
    inline const Eigen::Vector3d& meas() const {
      return measure_;
    }

    // setter
    inline Eigen::Vector3d& meas() {
      return measure_;
    }

  protected:
    Eigen::Vector3d measure_ = Eigen::Vector3d::Zero();
  };

  // odom observation model with offset
  class OdomObs {
  public:

    template <typename StateType>
    Eigen::Isometry3d const observation(const StateType& state) const {
      const Eigen::Matrix3d orientation = offset_.linear() * state.rotation();
      const Eigen::Vector3d pose        = offset_.linear() * state.position() + offset_.translation();

      Eigen::Isometry3d obs;
      obs.translation()  = pose;
      obs.linear()       = orientation;

      return obs;
    }

    Vector6d const obs_dist(const Eigen::Isometry3d& z_i) const {
      const Eigen::Matrix3d R_error = measure_.linear() * z_i.linear().transpose();

      Vector6d error;
      error.head(3) = measure_.translation() - z_i.translation();
      error.tail(3) = lie_algebra::SO3Log(R_error);
      return error;
    }

    //template <typename ObservationType>
    OdomObs const compute_mean_obs(const std::vector<OdomObs>& obs_vec){

      OdomObs z_bar;
      const int N = obs_vec.size();
      //std::cout<< "\n---[obs_debug] obs_vec.size() =" << obs_vec.size();
      Eigen::Vector3d t_mean = Eigen::Vector3d::Zero();

      // COMPUTING MEAN POSITION 
      for(const auto& obs : obs_vec) {
        t_mean += obs.meas().translation();
        //std::cout<< "\n---[obs_debug] t_mean=" << t_mean.transpose();
        //std::cout<< "\n---[obs_debug] obs.meas().translation()=" << obs.meas().translation().transpose();
      }
      t_mean /= static_cast<double>(N);
      //std::cout<< "\n---[obs_debug] t_mean/N =" << t_mean.transpose();

      // COMPUTING MEAN ROTATION with Rieman Itetative Mean
      Eigen::Matrix3d R_mean = obs_vec[0].meas().rotation();  // set the initial guess
      constexpr int max_iters = 100;
      constexpr double eps = 1e-9;
      for(int iter = 0; iter < max_iters; ++iter) {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();

        for(const auto& obs : obs_vec) {
          Eigen::Matrix3d R_i = obs.meas().rotation();
          Eigen::Matrix3d R_err = R_mean.transpose() * R_i;
          Eigen::AngleAxisd aa(R_err);
          Eigen::Vector3d phi = aa.angle() * aa.axis();
          delta += phi;
        }
        delta /= static_cast<double>(N);

        if(delta.norm() < eps) break;
        double angle = delta.norm();

        Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();

        if(angle > 1e-12) {
          Eigen::Vector3d axis = delta / angle;
          dR = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }

        // manifold update
        R_mean = R_mean * dR;
      }

      z_bar.meas().linear()      = R_mean;
      z_bar.meas().translation() = t_mean;

      return z_bar;
    }

    inline void setOffset(const Eigen::Isometry3d& offset) {
      offset_ = offset;
    }

    inline const size_t obsDim() const {
      // Tangent Space of Isometry3d
      return 6;
    }

    // getter
    inline const Eigen::Isometry3d& meas() const {
      return measure_;
    }

    // setter
    inline Eigen::Isometry3d& meas() {
      return measure_;
    }

  private:
    Eigen::Isometry3d measure_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d offset_  = Eigen::Isometry3d::Identity();
  };

} // namespace ukf_manifold