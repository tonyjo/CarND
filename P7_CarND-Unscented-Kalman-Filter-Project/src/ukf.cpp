#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Initial state vector
  x_ = VectorXd(5);

  // Initial covariance matrix with Identity matrix
  P_ = MatrixXd(5, 5);
  P_ << MatrixXd::Identity(5,5);

  // Time when the state is true, in us
  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; //Tune

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // Tune

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:
  Complete the initialization. See ukf.h for other member properties.
  Hint: one or more values initialized above might be wildly off...
  */

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight_1 = 0.5 / (n_aug_ + lambda_);

  weights_(0) = weight_0;
  for (int i = 1; i < (2 * n_aug_ + 1); i++) {
    weights_(i) = weight_1;
  }

  // Predicted sigma points matrix
  Xsig_pred_  = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // The current NIS for radar
  NIS_radar_ = 0;

  // The current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    TODO:
    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
      cout << "Kalman Filter Initialization " << endl;
      // state variables
      float px  = 0;
      float py  = 0;
      float v   = 0;
      float phi = 0;
      float phi_dot = 0;

      // Other variables
      float rho = 0;
      float rho_dot = 0;

      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Convert radar from polar to cartesian coordinates and initialize state.
        rho = meas_package.raw_measurements_[0];
        phi = meas_package.raw_measurements_[1];
        rho_dot = meas_package.raw_measurements_[2];

        px = rho * cos(phi);
        py = rho * sin(phi);

        if (fabs(px) < 0.001 && fabs(py)<0.001) {
          px = 0.001;
          py = 0.001;
        }

        x_ << px, py, v, phi, phi_dot;

      }else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        // Initialize state.
        px = meas_package.raw_measurements_[0];
        py = meas_package.raw_measurements_[1];

        if (fabs(px) < 0.001 && fabs(py)<0.001) {
          px = 0.001;
          py = 0.001;
        }

        x_ << px, py, v, phi, phi_dot;
      }

      // Done initializing,
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
      cout << "Done Initialization" << endl;
      // No need to predict or update
      return;
    }

    /**************************************************************************
    *  Prediction
    ***************************************************************************/
    // Compute the time elapsed between the current and previous measurements
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    // dt - expressed in seconds
    time_us_ = meas_package.timestamp_;
    // Prediction
    Prediction(dt);

    /**************************************************************************
    *  Update
    ***************************************************************************/
    // Skip update if input is all zeros
    if (meas_package.raw_measurements_[0] == 0 && meas_package.raw_measurements_[1] == 0){
      return;
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == true){
      // Update radar
      UpdateRadar(meas_package);

   }else if (meas_package.sensor_type_ == MeasurementPackage::LASER  && use_laser_ == true){
      // Update lidar
      UpdateLidar(meas_package);
  }

}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /*****************************************************************************
   * Generate Sigma points
   ****************************************************************************/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix for P_aug
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  /*****************************************************************************
   * Predict sigma points
   ****************************************************************************/

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // Predicted state values
    double px_p   = 0;
    double py_p   = 0;
    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * (sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    // Add noise
    px_p   = px_p   + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p   = py_p   + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p    = v_p    + nu_a * delta_t;
    yaw_p  = yaw_p  + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /*****************************************************************************
   * Predicted Mean and Covariance
   ****************************************************************************/

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_  + (weights_(i) * Xsig_pred_.col(i));
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    if(x_diff(3) > M_PI){
      x_diff(3) = (int(x_diff(3) - M_PI)%int(2*M_PI)) - M_PI;
    }else if(x_diff(3) < (-M_PI)){
      x_diff(3) = (int(x_diff(3) + M_PI)%int(2*M_PI)) + M_PI;
    }

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar NIS.
  */

  /*************************************************
   * Measurment Prediction
   *************************************************/
  //set measurement dimension, lidar can measure px and py
  int n_z = 2;

  // New measurement
  float px  = 0;
  float py  = 0;

  VectorXd z = VectorXd(2);
  z.fill(0.0);
  // Get the measurement
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  // Measurment space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    if (fabs(p_x) < 0.001 || fabs(p_y) < 0.001) {
      p_x = 0.001;
      p_y = 0.001;
    }

    // measurement model
    Zsig(0,i) = p_x;  //px
    Zsig(1,i) = p_y;  //py
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_ * std_laspx_, 0,
          0, std_laspy_ * std_laspy_;

  S = S + R;

  /*************************************************
   * Measurment Update
   *************************************************/

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = atan2(sin(x_diff(3)),cos(x_diff(3)));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS value
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  /*
   ************************************************
   * Measurment Prediction
   ************************************************
   */

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // New measurement
  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];

  // Measurment space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    //r
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
    //phi
    Zsig(1,i) = atan2(p_y,p_x);
    //angle normalization
    if(Zsig(1) > M_PI){
      Zsig(1) = (int(Zsig(1) - M_PI)%int(2*M_PI)) - M_PI;
    }else if(Zsig(1) < (-M_PI)){
      Zsig(1) = (int(Zsig(1) + M_PI)%int(2*M_PI)) + M_PI;
    }
    //r_dot
    if(fabs(sqrt(p_x*p_x + p_y*p_y)) > 0.001) {
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
    } else {
      Zsig(2,i) = 0.0;
    }

  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  //angle normalization
  if(z_pred(1) > M_PI){
    z_pred(1) = (int(z_pred(1) - M_PI)%int(2*M_PI)) - M_PI;
  }else if(z_pred(1) < (-M_PI)){
    z_pred(1) = (int(z_pred(1) + M_PI)%int(2*M_PI)) + M_PI;
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0,std_radrd_ * std_radrd_;
  S = S + R;

  /*************************************************
   * Measurment Update
   *************************************************/

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    if(z_diff(1) > M_PI){
      z_diff(1) = z_diff(1) - (2*M_PI);
    }else if(z_diff(1) < (-M_PI)){
      z_diff(1) = z_diff(1) + (2*M_PI);
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    if(x_diff(3) > M_PI){
      x_diff(3) = x_diff(3) - (2*M_PI);
    }else if(x_diff(3) < (-M_PI)){
      x_diff(3) = x_diff(3) + (2*M_PI);
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //Angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS value
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
