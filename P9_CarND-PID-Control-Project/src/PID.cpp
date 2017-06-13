#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp1, double Ki1, double Kd1) {
  Kp = Kp1;
  Ki = Ki1;
  Kd = Kd1;
}

void PID::UpdateError(double cte) {
  // D-Controller Update
  d_error = cte - p_error;
  // P-Controller Update
  p_error = cte;
  // I-Controller Update
  i_error += cte;
}

double PID::TotalError() {
  return (Kp*p_error + Ki*i_error + Kd*d_error);
}
