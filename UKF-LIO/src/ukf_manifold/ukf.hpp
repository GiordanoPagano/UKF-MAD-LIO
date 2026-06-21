namespace ukf_manifold {
  // d dim state
  // q dim input
  template <class StateType, class ObservationType>
  void UKF_<StateType, ObservationType>::propagate(StateType& state,
                                                   const Eigen::VectorXd& input,
                                                   const Eigen::MatrixXd& CholQ,
                                                   const double& dt) {
    const int dim_state = state.stateDim(); // dim state
    const int dim_input = state.inputDim(); // dim state

    Eigen::MatrixXd tol_mat(dim_state, dim_state);
    tol_mat.setIdentity();
    state.covariance().noalias() += tol_ * tol_mat;

    // apply transition current state omitting noise
    StateType prev_state = state;
    Eigen::VectorXd zero_noise(dim_input);
    zero_noise.setZero();
    if constexpr (std::is_same_v<StateType, NavStateLidar>){
      Eigen::Vector3d acc       = input.head(3);
      Eigen::Vector3d gyro      = input.tail(3);
      Eigen::Vector3d no_noise  = Eigen::Vector3d::Zero();
      state.transition(acc, gyro, no_noise, no_noise, no_noise, no_noise, dt);
    }
    else {
      state.transition(input, zero_noise, dt);
    }

    // keep cov dimension
    Eigen::LLT<Eigen::MatrixXd> chol(state.covariance()); // compute the Cholesky decomposition of A
    const Eigen::MatrixXd& L  = chol.matrixL();
    const Eigen::MatrixXd xis = ws_.w_d.sqrt_lambda_ * L;

    Eigen::MatrixXd xis_new(dim_state, 2 * dim_state);
    xis_new.setZero();

    for (int j = 0; j < dim_state; ++j) {
      // apply perturbation to sigma points
      std::unique_ptr<StateType> s_j_p(prev_state.boxplus(xis.col(j)));
      std::unique_ptr<StateType> s_j_m(prev_state.boxplus(-xis.col(j)));
      // propagate with input
      if constexpr (std::is_same_v<StateType, NavStateLidar>){
        Eigen::Vector3d acc       = input.head(3);
        Eigen::Vector3d gyro      = input.tail(3);
        Eigen::Vector3d no_noise  = zero_noise.tail(3);
        s_j_p->transition(acc, gyro, no_noise, no_noise, no_noise, no_noise, dt);
        s_j_m->transition(acc, gyro, no_noise, no_noise, no_noise, no_noise, dt);
      }
      else {
        s_j_p->transition(input, zero_noise, dt);
        s_j_m->transition(input, zero_noise, dt);
      }
      // go back in R^d to calculate covariance
      xis_new.col(j)             = *state.boxminus(*s_j_p);
      xis_new.col(dim_state + j) = *state.boxminus(*s_j_m);
    }

    // compute covariance, xi_mean has size (dim_state, 1)
    const Eigen::VectorXd xi_mean = ws_.w_d.wj_ * xis_new.rowwise().sum();

    xis_new.colwise() -= xi_mean;
    state.covariance() = ws_.w_d.wj_ * (xis_new * xis_new.transpose()) + ws_.w_d.wc0_ * (xi_mean * xi_mean.transpose());

    // compute covariance wrt to input noise (dim_state, 2*dim_noise_input)
    // if using UKF_LIO -> (dim_state, 2*(dim_noise_input + dim_noise_bias_input))
    Eigen::MatrixXd xis_new_noise(dim_state, 2 * dim_input);
    xis_new_noise.setZero();
    for (int i = 0; i < dim_input; ++i) {
      const Eigen::VectorXd w_p = ws_.w_q.sqrt_lambda_ * CholQ.col(i); // chol decomposition of noise covariance
      const Eigen::VectorXd w_m = -w_p;
      StateType s_j_p_new       = prev_state;
      StateType s_j_m_new       = prev_state;
      if constexpr (std::is_same_v<StateType, NavStateLidar>){  // if using UKF_LIO, in CholQ there is the complete Q matrix: [Q_input, Q_bias]
        Eigen::Vector3d acc                  = input.head(3);
        Eigen::Vector3d gyro                 = input.tail(3);
        Eigen::Vector3d acc_noise            = w_p.head(3);
        Eigen::Vector3d gyro_noise           = w_p.segment(3,3);
        Eigen::Vector3d acc_bias_noise       = w_p.segment(6,3);
        Eigen::Vector3d gyro_bias_noise      = w_p.tail(3);
        s_j_p_new.transition(acc, gyro, acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise, dt);
        acc_noise                            = w_m.head(3);
        gyro_noise                           = w_m.segment(3,3);
        acc_bias_noise                       = w_m.segment(6,3);
        gyro_bias_noise                      = w_m.tail(3);
        s_j_m_new.transition(acc, gyro, acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise, dt);
      }
      else {
        s_j_p_new.transition(input, w_p, dt);
        s_j_m_new.transition(input, w_m, dt);
      }
      xis_new_noise.col(i)             = *state.boxminus(s_j_p_new);
      xis_new_noise.col(i + dim_input) = *state.boxminus(s_j_m_new);
    }

    // compute covariance wrt to noise
    const Eigen::VectorXd xi_mean_noise = ws_.w_q.wj_ * xis_new_noise.rowwise().sum();
    xis_new_noise.colwise() -= xi_mean_noise;
    const Eigen::MatrixXd Q =
      ws_.w_q.wj_ * (xis_new_noise * xis_new_noise.transpose()) + ws_.w_q.wc0_ * (xi_mean_noise * xi_mean_noise.transpose());
    // add contribution of noise covariance to overall covariance
    state.covariance().noalias() += Q;
  }

  template <class StateType, class ObservationType>
  Eigen::VectorXd UKF_<StateType, ObservationType>::update(StateType& state,
                                                           ObservationType& z,
                                                           const Eigen::MatrixXd& R) {
    const int dim_state = state.stateDim(); // dim state
    const int dim_obs   = z.obsDim();

    Eigen::MatrixXd tol_mat(dim_state, dim_state);
    tol_mat.setIdentity();
    //    Vediamo cosa succede le la matrice di covarianza non viene toccata............. 
    // PS: non succede nulla ...............
    //state.covariance().noalias() += tol_ * tol_mat;

    //  TODO move R covariance measurement out this all

    // compute the Cholesky decomposition of A
    Eigen::LLT<Eigen::MatrixXd> chol(state.covariance());
    const Eigen::MatrixXd& L = chol.matrixL();
    // set sigma points
    const Eigen::MatrixXd xis = ws_.w_u.sqrt_lambda_ * L;

    std::vector<ObservationType>  zs;
    zs.resize(2 * dim_state);
    //zs.setZero();
    ObservationType z_hat;
    z_hat.meas() = z.observation(state);

    for (int j = 0; j < dim_state; ++j) {
      // apply perturbation to sigma points
      std::unique_ptr<StateType> chi_j_plus(state.boxplus(xis.col(j)));
      std::unique_ptr<StateType> chi_j_minus(state.boxplus(-xis.col(j)));
      // propagate through observation
      zs[j].meas()             = z.observation(*chi_j_plus);
      zs[dim_state + j].meas() = z.observation(*chi_j_minus);
    }

    // dim (dim_obs, 1)
    //Eigen::VectorXd z_bar = ws_.w_u.wm0_ * z_hat + ws_.w_u.wj_ * zs.rowwise().sum();
    // Computing the mean values in manifold for observation
    //if constexpr (std::is_same_v<ObservationType, OdomObs>){
    //  std::cout<< "\n---[obs_debug] zs.size() = " << zs.size();
    //  std::cout<< "\n---[obs_debug] zs[0].meas().translation() = " << zs[0].meas().translation().transpose();
    //}
    ObservationType z_bar = z.compute_mean_obs(zs);

    // initializing data for observation covariance matrix Pzz
    Eigen::MatrixXd new_zs(dim_obs, 2 * dim_state);
    Eigen::VectorXd new_z_hat(dim_obs, 1);
    
    // prune mean before computing covariance
    //zs.colwise() -= z_bar;
    for(int i=0; i<2*dim_state; ++i) new_zs.col(i) = zs[i].obs_dist(z_bar.meas());
    new_z_hat = z_hat.obs_dist(z_bar.meas());

    // compute covariance and cross covariance matrices, dim (dim_obs, dim_obs)
    //Eigen::MatrixXd P_zz = ws_.w_u.wc0_ * (z_hat * z_hat.transpose()) + ws_.w_u.wj_ * (zs * zs.transpose());
    Eigen::MatrixXd P_zz = ws_.w_u.wc0_ * (new_z_hat * new_z_hat.transpose()) + ws_.w_u.wj_ * (new_zs * new_zs.transpose());
    P_zz.noalias() += R;

    Eigen::MatrixXd xis_stacked(dim_state, 2 * dim_state);
    xis_stacked.setZero();
    xis_stacked.leftCols(dim_state)  = xis;
    xis_stacked.rightCols(dim_state) = -xis;

    // dim P_xz = (dim_state, dim_obs)
    //Eigen::MatrixXd P_xz = ws_.w_u.wj_ * xis_stacked * zs.transpose();
    Eigen::MatrixXd P_xz = ws_.w_u.wj_ * xis_stacked * new_zs.transpose();
    // solve system and get Kalman gain dim (dim_state, dim_obs)
    const Eigen::MatrixXd K = P_xz * P_zz.inverse();
    // update state
    //const Eigen::VectorXd error   = meas - z_bar;
    const Eigen::VectorXd error   = z.obs_dist(z_bar.meas());
    const Eigen::VectorXd xi_plus = K * error;
    write_log("   ");
    write_log(error.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)));
    write_log("   ");
    write_log(xi_plus.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)));
    if constexpr (std::is_same_v<ObservationType, OdomObs>){
      write_log("   ");
      write_log(z_bar.meas().translation().transpose().format(Eigen::IOFormat(Eigen::FullPrecision)));
    }
    write_log("\n");
    StateType old_state           = state;
    state                         = *old_state.boxplus(xi_plus);

    //std::cout<<"\n xi_plus dimension: "<< xi_plus.size() <<" and old_state dimension: "<< old_state.stateDim() <<"\n";

    state.covariance().noalias() -= (K * P_zz * K.transpose());
    state.covariance() = 0.5 * (state.covariance() + state.covariance().transpose());
    return error;
  }

  template <class StateType, class ObservationType>
  Weight UKF_<StateType, ObservationType>::setWeight(const int n, const double alpha) {
    const double alpha2 = alpha * alpha;
    const double lambda = (alpha2 - 1.0) * n;
    Weight w;
    w.lambda_      = lambda;
    w.sqrt_lambda_ = sqrt(n + lambda);
    w.wj_          = 1.0 / (2.0 * (n + lambda));
    w.wm0_         = lambda / (lambda + n);
    w.wc0_         = lambda / (lambda + n) + 3.0 - alpha2;
    return w;
  }

} // namespace ukf_manifold