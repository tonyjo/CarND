#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse = VectorXd(4);
  rmse << 0,0,0,0;

  // Check if the estimations and ground_truth have same length
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    cout << "Invalid estimations data or ground truth data" << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for(unsigned int i = 0; i < estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();

    rmse +=  residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root of the mean
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}
