#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

typedef uint_fast32_t uintf32;

class DMPair {

public:
  const std::string sample;
  std::vector<uintf32> profile;

  DMPair(const std::string name, std::vector<uintf32> prof)
      : sample(name), profile(std::move(prof)) {}

  DMPair(DMPair &&other) noexcept
      : sample(other.sample), profile(other.profile) {}

  DMPair(const DMPair &other) : sample(other.sample), profile(other.profile) {}

  ~DMPair() {}

  friend std::ostream &operator<<(std::ostream &os, const DMPair &obj) {

    os << obj.sample << ": ";
    for (auto i : obj.profile) {
      os << "\t" << i;
    }
    os << std::endl;
    return os;
  }
};

uintf32 hamming_distance(const DMPair &p1, const DMPair &p2) {
  uintf32 dist = 0;
  for (size_t i = 0; i < p1.profile.size(); i++) {
    if ((p1.profile[i] != p2.profile[i]) && p1.profile[i] != 0 &&
        p2.profile[i] != 0) {
      dist++;
    }
  }
  return dist;
}

void clear_memory(std::shared_ptr<std::vector<DMPair>> data) {
  for (auto &profile : *data) {
    std::vector<uintf32> tmp(0);
    profile.profile.swap(tmp);
  }
}

int main(int argc, char *argv[]) {

  // Get Profiles
  std::vector<DMPair> profile_data;
  std::ifstream inputFile(argv[1]);
  if (inputFile.is_open()) {
    std::string line; // Storage for profile
    if (inputFile.is_open()) {
      std::string header;
      std::getline(inputFile, header);
      auto columns = std::count(header.begin(), header.end(), '\t');
      while (std::getline(inputFile, line)) {
        std::istringstream tokens(line);
        std::string code;
        std::string sample;
        std::getline(tokens, sample, '\t');
        std::vector<uintf32> profile(columns);
        size_t idx = 0;
        while (std::getline(tokens, code, '\t')) {
          // Convert stuff to int if possible
          uintf32 allele = 0;
          try {
            allele = (uintf32)std::stol(code);
          } catch (std::invalid_argument const &ex) {
            // Unhandled as allele will be set to 0 anyways
          }
          profile[idx] = allele;
          idx++;
        }
        DMPair new_sample(sample, std::move(profile));
        profile_data.push_back(new_sample);
      }
      inputFile.close();
    } else {
      std::cerr << "Could not open file: " << argv[1] << std::endl;
    }
  }

  // Divide the range into some amount of cores and call this function in each
  // thread No need for a mutex
  std::vector<uintf32> output_matrix(profile_data.size() * profile_data.size());
  // Populate distance matrix between profiles
  for (size_t i = 0; i < profile_data.size(); i++) {
    for (size_t f = i; f < profile_data.size(); f++) {
      uintf32 dist = hamming_distance(profile_data[i], profile_data[f]);
      output_matrix[(i * profile_data.size()) + f] = dist;
      output_matrix[(f * profile_data.size()) + i] = dist;
    }
  }

  auto shared_data = std::make_shared<std::vector<DMPair>>(profile_data);
  std::thread clear_profiles(clear_memory, shared_data);

  std::cout << "Sample" << "\t";
  for (const DMPair &d : profile_data) {
    std::cout << d.sample << "\t";
  }
  std::cout << "\n";
  std::cout << profile_data[0].sample << "\t";

  size_t idx = 0;
  for (size_t i = 0; i < output_matrix.size(); i++) {
    std::cout << output_matrix[i] << "\t";
    uintf32 mod = (i + 1) % profile_data.size();
    if (mod == 0) {
      idx++;
      std::cout << '\n';
      std::cout << profile_data[idx].sample << "\t";
    }
  }
  clear_profiles.join();

  return 0;
}
