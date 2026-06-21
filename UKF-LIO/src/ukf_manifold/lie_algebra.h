
#pragma once
#include <Eigen/Core>

namespace ukf_manifold {
  template <typename MatrixType_>
  void fixRotation(MatrixType_& R) {
    MatrixType_ E = R.transpose() * R;
    E.diagonal().array() -= 1;
    R -= 0.5 * R * E;
  }
} // namespace ukf_manifold

namespace lie_algebra {

  //! skew symmetric matrix
  inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.;
    return S;
  }

  inline Eigen::Matrix3d SO3Exp(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d R;
    const double theta_square = omega.dot(omega);
    const double theta        = sqrt(theta_square);
    Eigen::Matrix3d W         = skew(omega);
    Eigen::Matrix3d K         = W / theta;
    if (theta_square < 1e-8) {
      R = Eigen::Matrix3d::Identity() + W;
    } else {
      double one_minus_cos = 2.0 * sin(theta / 2.0) * sin(theta / 2.0);
      R                    = Eigen::Matrix3d::Identity() + sin(theta) * K + one_minus_cos * K * K;
    }
    return R;
  }

  inline Eigen::Matrix3d SO3_rightJacobian(const Eigen::Vector3d& phi){
    const double angle = phi.norm();
    if (angle < 1e-10) return Eigen::Matrix3d::Identity() - 0.5 * skew(phi);

    const Eigen::Matrix3d S = skew(phi);
    return Eigen::Matrix3d::Identity() - (1.0 - std::cos(angle)) / (angle*angle) * S + (angle - std::sin(angle)) / (angle*angle*angle) * S * S;
  }

  inline Eigen::Vector3d SO3Log(const Eigen::Matrix3d& R) {
    const double &R11 = R(0, 0), R12 = R(0, 1), R13 = R(0, 2);
    const double &R21 = R(1, 0), R22 = R(1, 1), R23 = R(1, 2);
    const double &R31 = R(2, 0), R32 = R(2, 1), R33 = R(2, 2);

    // Get trace(R)
    const double tr = R.trace();
    const double pi(M_PI);
    const double two(2);

    Eigen::Vector3d omega;
    // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    // we do something special
    if (tr + 1.0 < 1e-10) {
      if (abs(R33 + 1.0) > 1e-5) {
        omega = (pi / sqrt(two + two * R33)) * Eigen::Vector3d(R13, R23, 1.0 + R33);
      } else if (abs(R22 + 1.0) > 1e-5) {
        omega = (pi / sqrt(two + two * R22)) * Eigen::Vector3d(R12, 1.0 + R22, R32);
      } else {
        omega = (pi / sqrt(two + two * R11)) * Eigen::Vector3d(1.0 + R11, R21, R31);
      }
    } else {
      double magnitude;
      const double tr_3 = tr - 3.0; // always negative
      if (tr_3 < -1e-7) {
        double theta = acos((tr - 1.0) / two);
        magnitude    = theta / (two * sin(theta));
      } else {
        // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
        // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
        magnitude = 0.5 - tr_3 * tr_3 / 12.0;
      }
      omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
    }
    return omega;
  }

} // namespace lie_algebra