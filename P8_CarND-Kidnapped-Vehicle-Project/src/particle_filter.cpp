/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author 1: Tiffany Huang
 *      Author 2: Tony Joseph         
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::default_random_engine gen;

	num_particles = 5;

	// Create a normal (Gaussian) distribution for x,y and theta.
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	double sample_x = 0;
	double sample_y = 0;
	double sample_theta = 0;

	for(int i = 0; i < num_particles; i++){
		// Sample  and from these normal distrubtions
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		particles.push_back({i, sample_x, sample_y, sample_theta, 1.0});
	}

	weights.resize(num_particles);
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;

	double std_x     = std_pos[0];
	double std_y     = std_pos[1];
	double std_theta = std_pos[2];


	for(int i = 0; i < particles.size(); i++){
		if (fabs(yaw_rate) < 1e-6){
			particles[i].x += (velocity * delta_t * cos(particles[i].theta));
			particles[i].y += (velocity * delta_t * sin(particles[i].theta));
			particles[i].theta = particles[i].theta;

		} else{
			particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta)- cos(particles[i].theta + (yaw_rate*delta_t)));
			particles[i].theta += yaw_rate*delta_t;
		}

		// Create a normal (Gaussian) distribution for x,y and theta.
		std::normal_distribution<double> dist_x(particles[i].x, std_x);
		std::normal_distribution<double> dist_y(particles[i].y, std_y);
		std::normal_distribution<double> dist_theta(particles[i].theta, std_theta);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	double x1 = 0;
	double y1 = 0;
	double x2 = 0;
	double y2 = 0;
	double min_distance = 0;
	double closest_id = 0;

	for(int i = 0; i < observations.size(); i++){
		min_distance = 999999.0; // Assign a large number
		closest_id = 0;

		x1 = observations[i].x;
		y1 = observations[i].y;

		for(int j = 0; j < predicted.size(); j++){
			x2 = predicted[j].x;
			y2 = predicted[j].y;

			// Compute Euclidean Distance
			double distance = dist(x1, y1, x2, y2);

			if (distance < min_distance){
				min_distance = distance;
				closest_id = j;
			}
		}

		// Assign the observed measurement to this particular landmark
		observations[i].id = predicted[closest_id].id;
		observations[i].x  = predicted[closest_id].x;
		observations[i].y  = predicted[closest_id].y;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	//
	// standard deviation
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	// Variables
	float converted_x  = 0;
	float converted_y  = 0;
	double compute_range = 0;
	double min_dist = 9999999.0;
	float mgd1;
	float mgd2;
	double mean_x;
	double mean_y;

	// For each particle compute the weights based on observations
	for(int i = 0; i < particles.size(); i++){
		particles[i].weight = 1.0; // Seting the weights to 1
		for(int j = 0; j < observations.size(); j++){
			// Conversion from vechile to map
			double converted_x = particles[i].x + (observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta));
			double converted_y = particles[i].y + (observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta));
			double min_dist = 9999999.0;

			// Find associated landmark
			for(int k = 0; k < map_landmarks.landmark_list.size(); k++){
				double x2 = map_landmarks.landmark_list[k].x_f;
				double y2 = map_landmarks.landmark_list[k].y_f;
				compute_range = dist(converted_x, converted_y, x2, y2);
				// Check for the closest associated landmark
				if(compute_range <= sensor_range && compute_range < min_dist){
					min_dist = compute_range;
					mean_x = x2;
					mean_y = y2;
				}
			}

			// Update weights
			mgd1 = ((converted_x - mean_x)*(converted_x - mean_x))/(2.0*std_x*std_x);
			mgd2 = ((converted_y - mean_y)*(converted_y - mean_y))/(2.0*std_y*std_y);
			particles[i].weight *= (1/(2.0*M_PI*std_x*std_y))*(exp(-1.0*(mgd1 + mgd2)));

		}
		// Update the newly compute particle weights in weights
		weights[i] = particles[i].weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Set of resampled particles
	std::vector<Particle> resampled_particles;

	std::random_device rnd;
	std::mt19937 gen(rnd());
	std::discrete_distribution<int> distribution{weights.begin(), weights.end()};

	// Resample particles with replacement with probability proportional to weight.
	for (int i=0; i < particles.size(); i++) {
		int random_particle = distribution(gen);

		// Sample  and from these normal distrubtions
		double resample_x = particles[random_particle].x;
		double resample_y = particles[random_particle].y;
		double resample_theta  = particles[random_particle].theta;
		double resample_weight = particles[random_particle].weight;

		resampled_particles.push_back({i, resample_x, resample_y, resample_theta, resample_weight});
	}

	//Update the particles
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
