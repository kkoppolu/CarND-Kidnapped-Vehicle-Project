/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

namespace {
const size_t Num_Particles = 100;
}
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = Num_Particles;

  // Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  particles.reserve(num_particles);
  int id = 0;
  for (int i = 0; i < num_particles; ++i) {
    // initialize each particle
    Particle p;
    p.id = id++;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    // add it to our list of weights
    weights.push_back(p.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // std::cout << "Predicting" << std::endl;
  // TODO: Add measurements to each particle and add random Gaussian noise.
  std::default_random_engine gen;

  // x_f = x_0 + v_0(sin(theta_0 + theta_dot * dt) - sin(theta_0)/theta_dot
  // y_f = y_0 + v_0(cos(theta_0) - cos(theta_0 + theta_dot * dt)/theta_dot
  double theta_mean, x_mean, y_mean;
  for (int i = 0; i < particles.size(); ++i) {
    if (yaw_rate == 0) {
      theta_mean = particles[i].theta;
      x_mean = particles[i].x + (velocity * delta_t * cos(theta_mean));
      y_mean = particles[i].y + (velocity * delta_t * sin(theta_mean));
    } else {
      theta_mean = particles[i].theta + (yaw_rate * delta_t);
      x_mean = particles[i].x + (velocity / yaw_rate) *
                                    (sin(theta_mean) - sin(particles[i].theta));
      y_mean = particles[i].y + (velocity / yaw_rate) *
                                    (cos(particles[i].theta) - cos(theta_mean));
    }

    normal_distribution<double> dist_x(x_mean, std_pos[0]);
    normal_distribution<double> dist_y(y_mean, std_pos[1]);
    normal_distribution<double> dist_theta(theta_mean, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  for (int i = 0; i < observations.size(); ++i) {
    double minDist = std::numeric_limits<double>::max();
    for (auto pred : predicted) {
      double del_x = pred.x - observations[i].x;
      double del_y = pred.y - observations[i].y;
      double dist = std::sqrt((del_x * del_x) + (del_y * del_y));
      if (dist < minDist) {
        // we have a new closest predicted landmark
        observations[i].id = pred.id;
        minDist = dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  if (mapLandmarksById.empty()) {
    for (auto l : map_landmarks.landmark_list) {
      mapLandmarksById[l.id_i] = l;
    }
  }

  double weightSum = 0;
  for (int i = 0; i < particles.size(); ++i) {
    // reset weight
    particles[i].weight = 1.0;

    std::vector<LandmarkObs> landmarksInRange;
    // find all the landmarks in range
    for (auto l : map_landmarks.landmark_list) {
      double del_x = l.x_f - particles[i].x;
      double del_y = l.y_f - particles[i].y;
      double dist = std::sqrt((del_x * del_x) + (del_y * del_y));
      if (sensor_range >= dist) {
        LandmarkObs predictedObs;
        predictedObs.x = l.x_f;
        predictedObs.y = l.y_f;
        predictedObs.id = l.id_i;
        landmarksInRange.push_back(predictedObs);
      }
    }

    // transform the measurements to map co-ordinates
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to
    //   implement (look at equation 3.33
    //   http://planning.cs.uiuc.edu/node99.html

    std::vector<LandmarkObs> transformedObservations(observations.begin(),
                                                     observations.end());
    double theta = particles[i].theta;
    for (int j = 0; j < observations.size(); ++j) {
      double x_c = observations[j].x;
      double y_c = observations[j].y;

      // x_m = x_p + x_c*cos(theta) - y_c*sin(theta) ...
      double x_m = particles[i].x + (x_c * cos(theta) - y_c * sin(theta));
      double y_m = particles[i].y + (x_c * sin(theta) + y_c * cos(theta));

      transformedObservations[j].x = x_m;
      transformedObservations[j].y = y_m;
    }

    // populate the landmark id for each measurement
    dataAssociation(landmarksInRange, transformedObservations);

    // calculate probabilities
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double C = 2 * M_PI * sig_x * sig_y;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (auto o : transformedObservations) {
      std::unordered_map<int, Map::single_landmark_s>::const_iterator it =
          mapLandmarksById.find(o.id);
      if (it == mapLandmarksById.end()) {
        std::cerr << "No associated landmark found for measurement " << o.x
                  << "," << o.y << std::endl;
      } else {
        double mu_x = it->second.x_f;
        double mu_y = it->second.y_f;
        double del_x = o.x - mu_x;
        double del_y = o.y - mu_y;
        double exponent = (del_x * del_x) / (2 * sig_x * sig_x) +
                          (del_y * del_y) / (2 * sig_y * sig_y);
        double weightComponent = std::exp(-exponent) / C;
        particles[i].weight = particles[i].weight * weightComponent;

        associations.push_back(it->first);
        sense_x.push_back(o.x);
        sense_y.push_back(o.y);
      }
    }
    particles[i] =
        SetAssociations(particles[i], associations, sense_x, sense_y);

    weights[i] = particles[i].weight;
    weightSum += particles[i].weight;
  }  // for

  if (weightSum > 0 == false) {
    std::cerr << "Weight sum is not greater than 0: " << weightSum << std::endl;
    double uniformWeight = 1.0 / particles.size();
    for (int i = 0; i < particles.size(); ++i) {
      weights[i] = uniformWeight;
    }
  }
}

void ParticleFilter::resample() {
  std::default_random_engine gen;
  std::discrete_distribution<int> d(weights.begin(), weights.end());

  const int resampleSz = particles.size();
  for (int i = 0; i < resampleSz; ++i) {
    int idx = d(gen);
    Particle srcParticle = particles[idx];
    particles[i].x = srcParticle.x;
    particles[i].y = srcParticle.y;
    particles[i].theta = srcParticle.theta;
    particles[i].associations = srcParticle.associations;
    particles[i].sense_x = srcParticle.sense_x;
    particles[i].sense_y = srcParticle.sense_y;
  }
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
