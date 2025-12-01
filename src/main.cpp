#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

typedef struct Option {
  option long_opt;
  std::string help;
} Option;

const size_t MISSING_VALUE = 0;

typedef uint_fast32_t uint32f;

union Output {
  float scaled;
  uint32f hamming;
};

std::vector<size_t> sample_ranges(size_t profiles, size_t threads) {
  size_t samples_bin = profiles / threads;
  std::vector<size_t> bins;
  for (size_t i = 0; i < profiles; i = i + samples_bin) {
    bins.push_back(i);
  }

  return bins;
}

class DMPair {

public:
  const std::string sample;
  std::vector<size_t> profile;

  DMPair(const std::string name, std::vector<size_t> prof)
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

Output hamming_distance(const DMPair &p1, const DMPair &p2, const bool scaled,
                        const bool count_missing) {
  Output dist_out;
  uint32f dist = 0;
  uint32f compared_sites = p1.profile.size();

  if (count_missing) {
    for (size_t i = 0; i < p1.profile.size(); i++) {
      if (p1.profile[i] != p2.profile[i]) {
        dist++;
      }
    }
  } else {
    compared_sites = 0;
    for (size_t i = 0; i < p1.profile.size(); i++) {
      bool missing =
          (p1.profile[i] == MISSING_VALUE) || (p2.profile[i] == MISSING_VALUE)
              ? true
              : false;
      if (missing) {
        continue;
      }
      if (p1.profile[i] != p2.profile[i]) {
        dist++;
      }
      compared_sites++;
    }
  }

  dist_out.hamming = dist;
  if (scaled) {
    dist_out.scaled = ((float)dist / (float)compared_sites) * 100.0f;
  }

  return dist_out;
}

void clear_memory(std::vector<DMPair> &data) {
  for (auto &profile : data) {
    std::vector<size_t> tmp(0);
    profile.profile.swap(tmp);
  }
}

void populate_dist_matrix(size_t start, size_t end, size_t pdata_size,
                          const bool scaled, const bool count_missing,
                          std::vector<DMPair> &profile_data,
                          std::vector<Output> &output_matrix) {

  for (size_t i = start; i < end; i++) {
    for (size_t f = i; f < pdata_size; f++) {
      Output dist_out = hamming_distance(profile_data[i], profile_data[f],
                                         scaled, count_missing);
      output_matrix[(i * pdata_size) + f] = dist_out;
      output_matrix[(f * pdata_size) + i] = dist_out;
    }
  }
}

void write_scaled(std::vector<Output> &output_matrix,
                  std::vector<DMPair> &profiles) {

  std::cout << "dists" << "\t";
  for (const DMPair &d : profiles) {
    std::cout << d.sample << "\t";
  }
  std::cout << "\n";
  std::cout << profiles[0].sample << "\t";

  size_t idx = 0;
  for (size_t i = 0; i < output_matrix.size(); i++) {
    std::cout << output_matrix[i].scaled << "\t";
    size_t mod = (i + 1) % profiles.size();
    if (mod == 0) {
      idx++;
      std::cout << '\n';
      std::cout << profiles[idx].sample << "\t";
    }
  }
}

void write_hamming(std::vector<Output> &output_matrix,
                   std::vector<DMPair> &profiles) {

  std::cout << "dists" << "\t";
  for (const DMPair &d : profiles) {
    std::cout << d.sample << "\t";
  }
  std::cout << "\n";
  std::cout << profiles[0].sample << "\t";

  size_t idx = 0;
  for (size_t i = 0; i < output_matrix.size(); i++) {
    std::cout << output_matrix[i].hamming << "\t";
    size_t mod = (i + 1) % profiles.size();
    if (mod == 0) {
      idx++;
      std::cout << '\n';
      std::cout << profiles[idx].sample << "\t";
    }
  }
}

const Option long_opts[] = {
    {{"input", required_argument, 0, 'i'}, "Input file file of profiles."},
    {{"threads", optional_argument, 0, 't'},
     "How many threads to run. default = 1"},
    {{"missing", optional_argument, 0, 'm'},
     "Specify the charactar to use for missing values. default = 0"},
    {{"delimiter", optional_argument, 0, 'd'},
     "Delimiter for table. default = \\t"},
    {{"scaled", no_argument, 0, 's'}, "Calculate a scaled distance metric."},
    {{"count-missing", no_argument, 0, 'c'},
     "Include missing values in count of differences."}};

void print_help() {
  for (const Option &opt : long_opts) {
    std::cout << " --" << opt.long_opt.name << "| -" << (char)opt.long_opt.val
              << ": " << std::endl;
    std::cout << "\t" << opt.help;
    switch (opt.long_opt.has_arg) {
    case required_argument:
      std::cout << " [required]";
      break;
    case no_argument:
      std::cout << " [flag]";
      break;
    case optional_argument:
      std::cout << " [optional]";
      break;
    default:
      break;
    }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {

  int c = 0;
  const option long_options[] = {long_opts[0].long_opt, long_opts[1].long_opt,
                                 long_opts[2].long_opt, long_opts[3].long_opt,
                                 long_opts[4].long_opt, long_opts[5].long_opt,
                                 {0, 0, 0, 0}};

  const char *input_file = nullptr;
  uint8_t threads = 1;
  std::string zero_value = "0";
  char delimiter = '\t';
  bool scaled = false;
  bool count_missing = false;

  if (argc <= 1) {
    std::cout << "No args passed" << std::endl;
    exit(EXIT_FAILURE);
  }

  while (1) {

    int option_index = 0;
    c = getopt_long(argc, argv, "hi:t:m:d:sc", long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
    case 'i':
      input_file = optarg;
      break;
    case 't':
      threads = (uint8_t)std::stoi(std::string(optarg));
      break;
    case 's':
      scaled = true;
      break;
    case 'c':
      count_missing = true;
      break;
    case 'm':
      zero_value = optarg;
      break;
    case 'd':
      delimiter = *optarg;
      break;
    case 'h':
      print_help();
      exit(EXIT_SUCCESS);
      break;
    case '?':
      print_help();
      exit(EXIT_FAILURE);
      break;
    case ':':
      print_help();
      exit(EXIT_FAILURE);
    default:
      print_help();
      exit(EXIT_FAILURE);
    }
  }

  if (input_file == nullptr) {
    print_help();
    exit(EXIT_FAILURE);
  }
  // Get Profiles
  std::vector<DMPair> profile_data;

  std::ifstream inputFile(input_file);
  if (inputFile.is_open()) {
    std::string line; // Storage for profile
    if (inputFile.is_open()) {
      std::string header;
      std::getline(inputFile, header);
      auto columns = std::count(header.begin(), header.end(), delimiter);
      while (std::getline(inputFile, line)) {
        std::istringstream tokens(line);
        std::string code;
        std::string sample;
        std::getline(tokens, sample, delimiter);
        std::vector<size_t> profile(columns);
        size_t idx = 0;
        while (std::getline(tokens, code, delimiter)) {

          if (code == zero_value) {
            profile[idx] = MISSING_VALUE;
          } else {
            profile[idx] = std::hash<std::string>{}(code);
          }
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

  // Evenly space the profiles so each thread can get a bundle of profiles to
  // process they can then all write to the output matrix
  std::vector<size_t> ranges;
  if (threads <= 1) {
    threads = 1;
    ranges.push_back(0);
    ranges.push_back(profile_data.size());
  } else {
    ranges = sample_ranges(profile_data.size(), threads);
  }

  std::vector<std::thread> pool;

  // Can save memory making this the upper triangle array only.
  std::vector<Output> output_matrix(profile_data.size() * profile_data.size());

  for (size_t i = 0; i < ranges.size() - 1; i++) {
    pool.push_back(std::thread(populate_dist_matrix, ranges[i], ranges[i + 1],
                               profile_data.size(), scaled, count_missing,
                               std::ref(profile_data),
                               std::ref(output_matrix)));
  }

  // Join all threads
  for (std::thread &th : pool) {
    th.join();
  }

  std::thread clear_profiles(clear_memory, std::ref(profile_data));
  if (scaled) {
    write_scaled(output_matrix, profile_data);
  } else {
    write_hamming(output_matrix, profile_data);
  }

  clear_profiles.join();

  return 0;
}
