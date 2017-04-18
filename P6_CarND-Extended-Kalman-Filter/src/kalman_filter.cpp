#include "kalman_filter.h"
#include "math.h"
#include <iostream>
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  // Pre-calculate P*H because it is executed twice.
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt_ =  P_ * Ht;

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd S = H_ * PHt_ + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt_ * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

  // Map from catersian to polar.

  VectorXd h_(3);
  float px = x_(0);
  float py = x_(1);
  float pxpy = 0;
  float phi = 0;

  pxpy = px*px + py*py;
  phi  = atan2(py, px);

  if(phi > M_PI){
    h_ << sqrt(pxpy),
          (phi -(2*M_PI)),
          (((px * x_(2)) + (py * x_(3))) / sqrt(pxpy));
  }else if(phi < (-M_PI)){
    h_ << sqrt(pxpy),
          (phi + (2*M_PI)),
          (((px * x_(2)) + (py * x_(3))) / sqrt(pxpy));
  }else{
    h_ << sqrt(pxpy),
          phi,
          (((px * x_(2)) + (py * x_(3))) / sqrt(pxpy));
  }

  // Pre-calculate P*H because it is executed twice.
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt_ =  P_ * Ht;

  VectorXd y = z - h_;
  MatrixXd S = H_ * PHt_  + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt_  * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
