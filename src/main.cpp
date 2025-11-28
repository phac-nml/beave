#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

typedef uint_fast32_t uintf32;

void print_help() {
  std::cout << "Missing args, --input|--threads" << std::endl;
}

std::vector<uintf32> sample_ranges(uintf32 profiles, uintf32 threads) {
  uintf32 samples_bin = profiles / threads;
  std::vector<uintf32> bins;
  for (uintf32 i = 0; i < profiles; i = i + samples_bin) {
    bins.push_back(i);
  }

  return bins;
}

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

void clear_memory(std::vector<DMPair> &data) {
  for (auto &profile : data) {
    std::vector<uintf32> tmp(0);
    profile.profile.swap(tmp);
  }
}

void populate_dist_matrix(uintf32 start, uintf32 end, size_t pdata_size,
                          std::vector<DMPair> &profile_data,
                          std::vector<uintf32> &output_matrix) {

  for (size_t i = start; i < end; i++) {
    for (size_t f = i; f < pdata_size; f++) {
      uintf32 dist = hamming_distance(profile_data[i], profile_data[f]);

      output_matrix[(i * pdata_size) + f] = dist;
      output_matrix[(f * pdata_size) + i] = dist;
    }
  }
}

int main(int argc, char *argv[]) {

  int c = 0;

  const option long_options[] = {{"input", required_argument, 0, 'i'},
                                 {"threads", required_argument, 0, 't'},
                                 {0, 0, 0, 0}};

  const char *input_file = nullptr;
  uint8_t threads = 1;

  if (argc <= 1) {
    std::cout << "No args passed" << std::endl;
    exit(EXIT_FAILURE);
  }
  while (1) {

    int option_index = 0;
    // Have GCC ignore my overriding of the options
    c = getopt_long(argc, argv, "hi:t:", long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
    case '?':
      break;
    case 'i':
      input_file = optarg;
      break;
    case 't':
      threads = (uint8_t)std::stoi(std::string(optarg));
      break;
    case 'h':
      break;
    default:
      print_help();
      exit(EXIT_FAILURE);
    }
  }

  // Get Profiles
  std::vector<DMPair> profile_data;

  std::ifstream inputFile(input_file);
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
      std::cerr << "Could not open file: " << input_file << std::endl;
    }
  }

  // Evenly space the profiels so each thread can get a bundle of profiles to
  // process they can then all write to the output matrix
  std::vector<uintf32> ranges;
  if (threads <= 1) {
    threads = 1;
    ranges.push_back(0);
    ranges.push_back(profile_data.size());
  } else {
    ranges = sample_ranges(profile_data.size(), threads);
  }

  std::vector<std::thread> pool;
  std::vector<uintf32> output_matrix(profile_data.size() * profile_data.size());

  for (size_t i = 0; i < ranges.size() - 1; i++) {
    pool.push_back(std::thread(populate_dist_matrix, ranges[i], ranges[i + 1],
                               profile_data.size(), std::ref(profile_data),
                               std::ref(output_matrix)));
  }

  // Join all threads
  for (std::thread &th : pool) {
    th.join();
  }

  std::thread clear_profiles(clear_memory, std::ref(profile_data));

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
