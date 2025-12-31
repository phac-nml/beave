#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <immintrin.h>
#include <iostream>
#include <smmintrin.h>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <string_view>
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

constexpr size_t MISSING_VALUE = 0;
constexpr size_t INITIAL_VEC_SIZE = 10000;

typedef uint32_t uint32f;

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
  const std::vector<size_t> profile;

  DMPair(const std::string name, std::vector<size_t> prof)
      : sample(name), profile(std::move(prof)) {}

  DMPair(DMPair &&other) noexcept
      : sample(std::move(other.sample)), profile(std::move(other.profile)) {}

  DMPair(const DMPair &other) : sample(other.sample), profile(other.profile) {}

  ~DMPair() = default;

  friend std::ostream &operator<<(std::ostream &os, const DMPair &obj) {

    os << obj.sample << ": ";
    for (auto i : obj.profile) {
      os << "\t" << i;
    }
    os << std::endl;
    return os;
  }
};

Output hamming_distance(const std::vector<size_t> &p1,
                        const std::vector<size_t> &p2, const bool scaled,
                        const bool count_missing) {
  Output dist_out;
  uint32f dist = 0;
  uint32f compared_sites = p1.size();
  const size_t *__restrict__ p1_data = p1.data();
  const size_t *__restrict__ p2_data = p2.data();

  if (count_missing) {
    for (size_t i = 0; i < p1.size(); i++) {
      if (p1_data[i] != p2_data[i]) {
        dist++;
      }
    }
  } else {
    compared_sites = 0;
    size_t i = 0;
    // Can likely compute the below using simd instructions loading in multiple
    // values. then slicing the value up
    // TODO validate the program can run on all cpus like this
#if defined __AVX2__ || defined __SSE2__
    if (p1.size() >= 4) {
      // Getting roll overs in digits
      __m256i vcount = _mm256_set1_epi64x(0);
      __m256i vcomp = _mm256_set1_epi64x(0);
      for (; i + 4 < p1.size(); i += 4) {
        // logic
        // const bool valid =
        //     (p1_data[i] != MISSING_VALUE) & (p2_data[i] != MISSING_VALUE);
        // compared_sites += valid;
        // dist += valid & (p1_data[i] != p2_data[i]);
        //  Load the vectors
        __m256i comp_vec = _mm256_set1_epi64x(0);

        __m256i dist_vec = _mm256_set1_epi64x(0);

        __m256i vec_a = _mm256_set_epi64x(p1_data[i], p1_data[i + 1],
                                          p1_data[i + 2], p1_data[i + 3]);
        __m256i vec_b = _mm256_set_epi64x(p2_data[i], p2_data[i + 1],
                                          p2_data[i + 2], p2_data[i + 3]);
        __m256i missing_vec = _mm256_set1_epi64x(0);
        // thes need to be neq comparisons but the operation, but the
        // instruciont only exists in AVX512. The exclamation mark may work?
        __m256i all_ones = _mm256_set1_epi64x(-1);
        __m256i a1_missing =
            _mm256_xor_si256(_mm256_cmpeq_epi64(vec_a, missing_vec), all_ones);

        __m256i a2_missing =
            _mm256_xor_si256(_mm256_cmpeq_epi64(vec_b, missing_vec), all_ones);
        __m256i valid = _mm256_and_si256(a1_missing, a2_missing);

        vcomp = _mm256_add_epi64(valid, comp_vec);
        __m256i dist_vec_tmp =
            _mm256_xor_si256(_mm256_cmpeq_epi64(vec_a, vec_b), all_ones);

        dist_vec = _mm256_and_si256(valid, dist_vec_tmp);
        vcount = _mm256_add_epi64(vcount, dist_vec);
      }
      // Need to do unpacking here
      // dist += static_cast<size_t>(_mm256_extract_epi64(vcount, 0));
      // dist += static_cast<size_t>(_mm256_extract_epi64(vcount, 1));
      // dist += static_cast<size_t>(_mm256_extract_epi64(vcount, 2));
      // dist += static_cast<size_t>(_mm256_extract_epi64(vcount, 3));
      dist += _mm256_extract_epi64(vcount, 0);
      dist += _mm256_extract_epi64(vcount, 1);
      dist += _mm256_extract_epi64(vcount, 2);
      dist += _mm256_extract_epi64(vcount, 3);

      compared_sites += _mm256_extract_epi64(vcomp, 0);
      compared_sites += _mm256_extract_epi64(vcomp, 1);
      compared_sites += _mm256_extract_epi64(vcomp, 2);
      compared_sites += _mm256_extract_epi64(vcomp, 3);
      // Need to have these wrap around for some reason, definately need to
      // figure out why...
      dist = 0 - dist;
      compared_sites = 0 - compared_sites;
    }
#endif
    for (; i < p1.size(); i++) {
      const bool valid =
          (p1_data[i] != MISSING_VALUE) & (p2_data[i] != MISSING_VALUE);
      compared_sites += valid;
      dist += valid & (p1_data[i] != p2_data[i]);
    }
  }

  dist_out.hamming = dist;
  if (scaled) {
    dist_out.scaled =
        (static_cast<float>(dist) / static_cast<float>(compared_sites)) *
        100.0f;
  }

  return dist_out;
}

void populate_dist_matrix(size_t start, size_t end, size_t pdata_size,
                          const bool scaled, const bool count_missing,
                          const std::vector<std::vector<size_t>> &profile_data,
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
                     const std::vector<std::string> &query_names,
                     const std::vector<std::vector<size_t>> &query_data) {
  std::ostringstream local_buffer;
  if (scaled) {
    for (size_t i = start; i < end; i++) {
      for (size_t f = 0; f < query_data.size(); f++) {
        Output dist_out = hamming_distance(query_data[i], query_data[f], scaled,
                                           count_missing);
        local_buffer << query_names[i] << "\t" << query_names[f] << "\t"
                     << std::format("{:.6f}", dist_out.scaled) << "\n";
      }
    }
  } else {
    for (size_t i = start; i < end; i++) {
      for (size_t f = 0; f < query_data.size(); f++) {
        Output dist_out = hamming_distance(query_data[i], query_data[f], scaled,
                                           count_missing);
        local_buffer << query_names[i] << "\t" << query_names[f] << "\t"
                     << dist_out.hamming << "\n";
      }
    }
  }
  std::osyncstream(std::cout) << local_buffer.str();
}

void write_scaled(std::vector<Output> &output_matrix,
                  std::vector<std::string> &profiles) {

  std::cout << "dists" << "\t";
  for (const std::string &d : profiles) {
    std::cout << d << "\t";
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx] << "\t";
    size_t i = mat_idx;
    for (; i < mat_idx + profiles.size(); i++) {
      std::cout << std::format("{:.6f}", output_matrix[i].scaled) << "\t";
    }
    mat_idx = i;
    ++idx;
  } while (idx < profiles.size());
}

void write_hamming(std::vector<Output> &output_matrix,
                   std::vector<std::string> &profiles) {
  std::cout << "dists" << "\t";
  for (const std::string &d : profiles) {
    std::cout << d << "\t";
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx] << "\t";
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

void print_parser_help() {
  std::ostringstream local_buffer;
  local_buffer << "Subcommands:\n";
  local_buffer << "\tmatrix - Create distance matrix with an input profile.\n";
  local_buffer << "\tfast-match - Compare a set of profiles to a set of query "
                  "profiles.\n";
  local_buffer << "\nExamples:\n";
  local_buffer << "dist-mat matrix -i profiles.tsv -t 4 -sc > output.tsv\n";
  local_buffer << "dist-mat fast-match -i qprofiles.tsv -r profiles.tsv -t 4 "
                  "-sc > output.tsv\n";
  std::cout << local_buffer.str();
}

void print_help() {
  print_parser_help();
  std::cout << "\n";
  std::cout << "Command Options\n\n";
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
std::string read_profiles(const char *file,
                          std::vector<std::string> &data_names,
                          std::vector<std::vector<size_t>> &data_profiles,
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
    // DMPair new_sample(std::move(sample), std::move(profile));
    // data.emplace_back(new_sample);
    data_names.emplace_back(std::move(sample));
    data_profiles.emplace_back(std::move(profile));
  }
  fo.close();
  if (data_names.size() != data_profiles.size()) {
    throw std::length_error(
        "number of profiles names does not match number of profiles ingested.");
  }
  return header;
}

// Evenly space the profiles so each thread can get a bundle of profiles to
// process they can then all write to the output matrix
std::vector<size_t> get_thread_ranges(size_t threads, size_t data_size) {
  std::vector<size_t> ranges;
  if (threads <= 1 || data_size <= threads) {
    ranges.push_back(0);
    ranges.push_back(data_size);
  } else {
    ranges = sample_ranges(data_size, threads);
    ranges.push_back(data_size);
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

  auto hide_option = [](char opt) {
    for (auto &o : long_opts) {
      if (o.long_opt.val == opt) {
        o.print = false;
        break;
      }
    }
  };

  if (argc <= 1) {
    std::cout << "No args passed" << std::endl;
    print_parser_help();
    exit(EXIT_FAILURE);
  }
  std::string mat = "matrix";
  std::string fast_match = "fast-match";

  std::string_view arg1(argv[1]);
  if (arg1 == mat) {
    hide_option('r');
  } else if (arg1 == fast_match) {
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
      try {

        int t = std::stoi(std::string(optarg));
        if (t < 1) {
          std::cerr << "Error: Threads must be greater than 1 \n.";
          exit(EXIT_FAILURE);
        }
        threads = t;
      } catch (const std::exception &e) {
        std::cerr << "Error: invalid thread count. \n";
        exit(EXIT_FAILURE);
      }
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
    std::vector<std::string> profile_names;
    std::vector<std::vector<size_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(input_file, profile_names, profiles, delimiter, zero_value);

    std::vector<size_t> ranges =
        get_thread_ranges(threads, profile_names.size());

    std::vector<std::thread> pool;

    // Can save memory making this the upper triangle array only.
    std::vector<Output> output_matrix(profile_names.size() *
                                      profile_names.size());

    for (size_t i = 0; i < ranges.size() - 1; i++) {
      pool.push_back(std::thread(populate_dist_matrix, ranges[i], ranges[i + 1],
                                 profiles.size(), scaled, count_missing,
                                 std::cref(profiles), std::ref(output_matrix)));
    }

    // Join all threads
    for (std::thread &th : pool) {
      th.join();
    }

    if (scaled) {
      write_scaled(output_matrix, profile_names);
    } else {
      write_hamming(output_matrix, profile_names);
    }

    return 0;
  } else if (program == FASTMATCH) {
    std::vector<std::string> query_names;
    std::vector<std::vector<size_t>> query_profiles;
    query_names.reserve(INITIAL_VEC_SIZE);
    query_profiles.reserve(INITIAL_VEC_SIZE);

    std::string input_header = read_profiles(
        input_file, query_names, query_profiles, delimiter, zero_value);

    size_t length_input = query_names.size();

    // Reusing the vector to combine the data
    std::string ref_header = read_profiles(
        reference_file, query_names, query_profiles, delimiter, zero_value);

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
                                 scaled, count_missing, std::cref(query_names),
                                 std::cref(query_profiles)));
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
