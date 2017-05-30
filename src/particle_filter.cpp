/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  double vel_yaw = velocity / yaw_rate;

  for (Particle& particle: particles) {
    double theta = particle.theta + yaw_rate * delta_t;
    double x = particle.x + vel_yaw * (sin(theta) - sin(particle.theta));
    double y = particle.y + vel_yaw * (cos(particle.theta) - cos(theta));

    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

int findClosest(const vector<LandmarkObs> &predicted, LandmarkObs &obs) {
  double closest_dist = INFINITY;
  int id_of_closest;
  for (auto& pred: predicted) {
    double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
    if (current_dist < closest_dist) {
      closest_dist = current_dist;
      id_of_closest = pred.id;
    }
  }
  return id_of_closest;
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (LandmarkObs& obs: observations) {
    obs.id = findClosest(predicted, obs);
  }
}

vector<LandmarkObs> getProcessedObservations(Particle &particle, vector<LandmarkObs> observations) {
  vector<LandmarkObs> processed_observations;
  for (LandmarkObs& obs: observations) {
    double cosTheta = cos(particle.theta);
    double sinTheta = sin(particle.theta);
    double xt = particle.x + obs.x * cosTheta - obs.y * sinTheta;
    double yt = particle.y + obs.x * sinTheta + obs.y * cosTheta;
    LandmarkObs observation = { -1, xt, yt };
    processed_observations.push_back(observation);
  }

  return processed_observations;
}

vector<LandmarkObs> getLandmarksInRange(double sensor_range, Particle & particle, Map map_landmarks) {
  vector<LandmarkObs> landmarkInRange;
  int id = 0;
  for (Map::single_landmark_s& singleLandmark: map_landmarks.landmark_list) {
    if (dist(singleLandmark.x_f, singleLandmark.y_f, particle.x, particle.y) <= sensor_range) {
      LandmarkObs landmark = { id++, singleLandmark.x_f, singleLandmark.y_f };
      landmarkInRange.push_back(landmark);
    }
  }

  return landmarkInRange;
}

double getWeight(Particle & particle, vector<LandmarkObs> car_observations, vector<LandmarkObs> landmarkInRange, double std_landmark[]) {
  double weight = 1.0;
  for (LandmarkObs& obs: car_observations) {
    LandmarkObs land_obs = landmarkInRange[obs.id];
    double p_x = pow(obs.x - land_obs.x, 2) / (2 * pow(std_landmark[0], 2));
    double p_y = pow(obs.y - land_obs.y, 2) / (2 * pow(std_landmark[1], 2));
    double power = -1.0 * (p_x + p_y);
    weight *= exp(power);
  }

  particle.weight = weight;
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

  double sum_w = 0.0;

  for (Particle& particle: particles) {
    // process observations so that they are easier to search
    vector<LandmarkObs> processedObservations = getProcessedObservations(particle, observations);
    vector<LandmarkObs> landmarkInRange = getLandmarksInRange(sensor_range, particle, map_landmarks);

    // for debugging purposes
    dataAssociation(landmarkInRange, processedObservations);
    // accumulate weight
    sum_w += getWeight(particle, processedObservations, landmarkInRange, std_landmark);
  }

  // update weights and particles
  for (int i = 0; i < num_particles; i++) {
    particles[i].weight /= sum_w * 2 * M_PI * std_landmark[0] * std_landmark[1];
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  std::vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[distribution(gen)]);
  }

  particles = new_particles;
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
