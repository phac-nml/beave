#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <syncstream>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

typedef struct Option {
  option long_opt;
  std::string help;
  bool print;
} Option;

enum Program { FASTMATCH, MATRIX };

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

void fast_match_func(size_t start, size_t end, const bool scaled,
                     const bool count_missing,
                     std::vector<DMPair> &query_data) {
  std::osyncstream bout(std::cout);
  if (scaled) {
    for (size_t i = start; i < end; i++) {
      for (size_t f = 0; f < query_data.size(); f++) {
        Output dist_out = hamming_distance(query_data[i], query_data[f], scaled,
                                           count_missing);
        bout << query_data[i].sample << "\t" << query_data[f].sample << "\t"
             << std::format("{:.6f}", dist_out.scaled) << "\n";
      }
    }
  } else {
    for (size_t i = start; i < end; i++) {
      for (size_t f = 0; f < query_data.size(); f++) {
        Output dist_out = hamming_distance(query_data[i], query_data[f], scaled,
                                           count_missing);
        bout << query_data[i].sample << "\t" << query_data[f].sample << "\t"
             << dist_out.hamming << "\n";
      }
    }
  }
}

void write_scaled(std::vector<Output> &output_matrix,
                  std::vector<DMPair> &profiles) {

  std::cout << "dists" << "\t";
  for (const DMPair &d : profiles) {
    std::cout << d.sample << "\t";
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx].sample << "\t";
    size_t i = mat_idx;
    for (; i < mat_idx + profiles.size(); i++) {
      std::cout << std::format("{:.6f}", output_matrix[i].scaled) << "\t";
    }
    mat_idx = i;
    ++idx;
  } while (idx < profiles.size());
}

void write_hamming(std::vector<Output> &output_matrix,
                   std::vector<DMPair> &profiles) {

  std::cout << "dists" << "\t";
  for (const DMPair &d : profiles) {
    std::cout << d.sample << "\t";
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx].sample << "\t";
    size_t i = mat_idx;
    for (; i < mat_idx + profiles.size(); i++) {
      std::cout << output_matrix[i].hamming << "\t";
    }
    mat_idx = i;
    ++idx;
  } while (idx < profiles.size());
}

Option long_opts[] = {
    {{"input", required_argument, 0, 'i'},
     "Input file file of profiles.",
     true},
    {{"reference", required_argument, 0, 'r'},
     "Reference profiles to use for fast matching.",
     true},
    {{"threads", optional_argument, 0, 't'},
     "How many threads to run. default = 1",
     true},
    {{"missing", optional_argument, 0, 'm'},
     "Specify the charactar to use for missing values. default = 0",
     true},
    {{"delimiter", optional_argument, 0, 'd'},
     "Delimiter for table. default = \\t",
     true},
    {{"scaled", no_argument, 0, 's'},
     "Calculate a scaled distance metric.",
     true},
    {{"count-missing", no_argument, 0, 'c'},
     "Include missing values in count of differences.",
     true},
};

void print_help() {
  for (const Option &opt : long_opts) {
    if (!opt.print) {
      continue;
    }
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

void print_parser_help() {
  std::cout << "matrix - Create distance matrix with an input profile."
            << std::endl;
  std::cout
      << "fast-match - Compare a set of profiles to a set of query profiles."
      << std::endl;
}

std::string read_profiles(const char *file, std::vector<DMPair> &data,
                          char delimiter, std::string zero_value) {
  std::ifstream fo(file);
  if (!fo.is_open()) {
    std::cerr << "Could not open " << file << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line; // Storage for profile
  std::string header;
  std::getline(fo, header);
  auto columns = std::count(header.begin(), header.end(), delimiter);
  while (std::getline(fo, line)) {
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
    data.push_back(new_sample);
  }
  fo.close();
  return header;
}

// Evenly space the profiles so each thread can get a bundle of profiles to
// process they can then all write to the output matrix
std::vector<size_t> get_thread_ranges(size_t threads, size_t data_size) {
  std::vector<size_t> ranges;
  if (threads <= 1 || data_size <= threads) {
    threads = 1;
    ranges.push_back(0);
    ranges.push_back(data_size);
  } else {
    ranges = sample_ranges(data_size, threads);
  }
  return ranges;
}

int main(int argc, char *argv[]) {

  const option long_options[] = {long_opts[0].long_opt, long_opts[1].long_opt,
                                 long_opts[2].long_opt, long_opts[3].long_opt,
                                 long_opts[4].long_opt, long_opts[5].long_opt,
                                 long_opts[6].long_opt, {0, 0, 0, 0}};

  const char *input_file = nullptr;
  const char *reference_file = nullptr;
  uint8_t threads = 1;
  std::string zero_value = "0";
  char delimiter = '\t';
  bool scaled = false;
  bool count_missing = false;
  Program program = MATRIX;

  if (argc <= 1) {
    std::cout << "No args passed" << std::endl;
    print_parser_help();
    exit(EXIT_FAILURE);
  }
  std::string mat = "matrix";
  std::string fast_match = "fast-match";

  int REFERENCE_OPT = 1;
  if (argv[1] == mat) {
    long_opts[REFERENCE_OPT].print = false;
  } else if (argv[1] == fast_match) {
    program = FASTMATCH;
  } else {
    print_parser_help();
    exit(EXIT_FAILURE);
  }

  int c = 0;
  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, "hi:t:r:m:d:sc", long_options, &option_index);
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
    case 'r':
      if (program != FASTMATCH) {
        std::cout << "Reference option passed, but matrix program selected."
                  << std::endl;
        print_parser_help();
        print_help();
        exit(EXIT_FAILURE);
      }
      reference_file = optarg;
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
  if (reference_file == nullptr && program == FASTMATCH) {
    print_help();
    exit(EXIT_FAILURE);
  }

  if (program == MATRIX) {

    // Get Profiles
    std::vector<DMPair> profile_data;

    // Discarding return value here on purpose
    read_profiles(input_file, profile_data, delimiter, zero_value);

    std::vector<size_t> ranges =
        get_thread_ranges(threads, profile_data.size());

    std::vector<std::thread> pool;

    // Can save memory making this the upper triangle array only.
    std::vector<Output> output_matrix(profile_data.size() *
                                      profile_data.size());

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
  } else if (program == FASTMATCH) {
    std::vector<DMPair> query_data;

    std::string input_header =
        read_profiles(input_file, query_data, delimiter, zero_value);

    size_t length_input = query_data.size();

    // Reusing the vector to combine the data
    std::string ref_header =
        read_profiles(reference_file, query_data, delimiter, zero_value);

    if (input_header != ref_header) {
      std::cerr
          << "Headers differ between input and reference profiles. Bailing out."
          << std::endl;
      exit(EXIT_FAILURE);
    }

    // Get the range of threads to use based on the length of the reference
    // queries
    std::vector<size_t> ranges = get_thread_ranges(threads, length_input);

    std::vector<std::thread> pool;
    for (size_t i = 0; i < ranges.size() - 1; i++) {
      pool.push_back(std::thread(fast_match_func, ranges[i], ranges[i + 1],
                                 scaled, count_missing, std::ref(query_data)));
    }

    for (std::thread &th : pool) {
      th.join();
    }

    return 0;
  } else {
    exit(EXIT_FAILURE);
  }
  return 0;
}
