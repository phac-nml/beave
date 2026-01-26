#include "main.hpp"

constexpr size_t MISSING_VALUE = 0;
constexpr size_t INITIAL_VEC_SIZE = 10000;
constexpr size_t MINIMUM_PROFILES = 2;

typedef struct Option {
  option long_opt;
  std::string help;
  bool print;
} Option;

enum Program { FASTMATCH, MATRIX };

/**
 * @brief Determine the sample ranges to be calculated based on
 * the number of threads used.
 *
 * @param profiles The number of profiles to be processed
 * @param threads the number of threads used by the program
 *
 * @return A vector of indexes containing the ranges of samples to be dispatched
 *
 * @details
 * The number of threads is handled externally by the program, therefore
 * a value of size 0 should never be passed to the threads argument. There
 * must be atleast 2 profiles for any comparison to take place as well.
 *
 * @usage
 * std::vector<size_t> bins = sample_rnages(profiles.size(), threads)
 */
std::vector<size_t> sample_ranges(size_t profiles, size_t threads) {
  if (threads < 1 || profiles < MINIMUM_PROFILES) {
    throw std::invalid_argument("Threads passed must be a positive integer, "
                                "and atleast 2 profiles must be passed.");
  }
  size_t samples_bin = profiles / threads;
  std::vector<size_t> bins;
  for (size_t i = 0; i < profiles; i = i + samples_bin) {
    bins.push_back(i);
  }
  bins.push_back(profiles);

  return bins;
}

/**
 * @brief The main driver function for calculating the hamming distance scaled
 * and unsacled.
 *
 * @param p1 Profile one for comparison
 * @param p2 Profile two used for comparison
 * @param scaled A boolean value determining if the hamming distance should be
 * scaled to the number of comparisons made.
 * @param count_missing A boolean value determining if missing values e.g. those
 * set to 0 should be counted as differences.
 *
 * @return The function returns a union type of `Output` too allow for the same
 * function to be used for both scaled and un-scaled distances.
 *
 * @details
 * This function is the main driver for determining the hamming distance, as it
 * is in a hot loop it has been written to allow for the aggressive optimization
 * by the compiler. SIMD instructions were hand rolled however using uint32_t
 * allowed for the compiler to SIMD optimize the logic of excluding missing
 * counted values. Resulting in speed ups, compiler options should be added to
 * allow for SIMD optimizations with AVX-512 instruction sets for CPUs that
 * offer them.
 *
 *
 * @usage
 * Output o.hamming = hamming_distance(std::vector<uint32_t>{1, 2, 3, 4},
 * std::vector<uint32_t>{1, 2, 3, 4}, false, true)
 *
 *
 */
Output hamming_distance(const std::vector<uint32_t> &p1,
                        const std::vector<uint32_t> &p2, const bool scaled,
                        const bool count_missing) {
  Output dist_out;
  uint32_t dist = 0;
  uint32_t compared_sites = p1.size();
  const uint32_t *__restrict__ p1_data = p1.data();
  const uint32_t *__restrict__ p2_data = p2.data();

  if (count_missing) {
    for (size_t i = 0; i < p1.size(); i++) {
      if (p1_data[i] != p2_data[i]) {
        dist++;
      }
    }
  } else {
    compared_sites = 0;
    // Hand rolled SIMD instructions for this, but switching to uin32_t allowed
    // the compiler to optimize this code.
    for (size_t i = 0; i < p1.size(); i++) {
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

/**
 * @brief Populate the output distance matrix.
 *
 * @param start Index to begin begin calculation of the profiles on.
 * @param end Index to halt cacluation of the profiles on.
 * @param pdata_size The size of the profiles being used.
 * @param scaled Boolean value inidcating whether a scaled distance should be
 * used.
 * @param count_missing Boolean value for deciding if missing values are counted
 * as differences.
 * @param profile_data A reference to the passed in profiles.
 * @param output_matrix A refernce to the matrix handling the final results.
 *
 * @return No return value the function works through side effects as the final
 * matrix is shared betweent threads.
 *
 * @details
 * This function is the main program responsible for calculating the final
 * results and the subsequent distance matrix. This function is meant to be
 * called from multiple threads at one time, therefore there is no return value.
 *
 * @usage
 * populate_dist_matrix(0, 100, profiles.size(), false, false, profiles,
 * output_matrix)
 */
void populate_dist_matrix(
    size_t start, size_t end, size_t pdata_size, const bool scaled,
    const bool count_missing,
    const std::vector<std::vector<uint32_t>> &profile_data,
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

/**
 * @brief Calculate pairwise distances between a subset of profiles and a group
 * of references.
 *
 * @param start The start index to begin profile calculations.
 * @param end The end set of profiles to match up too, e.g. the size of the
 * query set.
 * @param scaled Tell the program to print the scaled distance.
 * @param count_missing A boolean flag passed to `hamming_distance` which will
 * tell the program to count missing values as differences
 * @param query_names The query names to be printed along side the output
 * samples
 * @param query_data The query data to be matched against
 *
 * @return Returns no value, output stream is instead updated.
 *
 *
 * @details
 * This function runs in a multithreaded environment, relying on the osyncstream
 * from C++ to write too stdout. While stdout is considered thread safe
 * interlacing of outputs can occur. The actual for loop logic is duplicated in
 * order to select the correct output type from the union, without putting an if
 * statement in the hotloop that may not be optimized properly by the compiler.
 * Outputs will be redirected to stdout.
 *
 * Floats are trunacated by C++ so instead of 66.66666 being written
 * out, 66.666672 is
 *
 *
 * @usage
 * fast_match_func(0, 100, false, false, query_names, query_data);
 *
 */
void fast_match_func(size_t start, size_t end, const bool scaled,
                     const bool count_missing,
                     const std::vector<std::string> &query_names,
                     const std::vector<std::vector<uint32_t>> &query_data) {
  // TODO once tests and benchmarks are setup some kind of dynamic dispatch
  // for the outputs written outputs should be tested either through partial
  // functions or using std::variant
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

/**
 * @brief Write the final scaled distance matrix to stdout.
 *
 * @param output_matrix The calculated distance matrix.
 * @param profiles The labels associated with each output result.
 *
 * @return Writes to stdout.
 *
 * @details
 * This function writes the final distance matrix to standard output, the logic
 * for this function and the one for writing the hamming distance is identical
 * and will likely be refactored in the future. This function is called after
 * the `populate_dist_matrix` has been called.
 *
 * @usage
 *
 * write_scaled(output_matrix, profiles);
 *
 */
void write_scaled(std::vector<Output> &output_matrix,
                  std::vector<std::string> &profiles) {

  std::cout << "dists";
  for (const std::string &d : profiles) {
    std::cout << "\t" << d;
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx];
    size_t i = mat_idx;
    for (; i < mat_idx + profiles.size(); i++) {
      std::cout << "\t" << std::format("{:.6f}", output_matrix[i].scaled);
    }
    mat_idx = i;
    ++idx;
  } while (idx < profiles.size());
}

/**
 * @brief Write the final hamming distance matrix to stdout.
 *
 * @param output_matrix The calculated distance matrix.
 * @param profiles The labels associated with each output result.
 *
 * @return Writes to stdout.
 *
 * @details
 * This function writes the final distance matrix to standard output, the logic
 * for this function and the one for writing the scaled distance is identical
 * and will likely be refactored in the future. This function is called after
 * the `populate_dist_matrix` has been called.
 *
 * @usage
 *
 * write_scaled(output_matrix, profiles);
 *
 */
void write_hamming(std::vector<Output> &output_matrix,
                   std::vector<std::string> &profiles) {
  std::cout << "dists";
  for (const std::string &d : profiles) {
    std::cout << "\t" << d;
  }

  size_t idx = 0;
  size_t mat_idx = 0;
  do {
    std::cout << '\n' << profiles[idx];
    size_t i = mat_idx;
    for (; i < mat_idx + profiles.size(); i++) {
      std::cout << "\t" << output_matrix[i].hamming;
    }
    mat_idx = i;
    ++idx;
  } while (idx < profiles.size());
}

/**
 * @brief An array containing the options passed to the CLI parser.
 */
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

/**
 * @brief Top level help message for the parser to print.
 */
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

/**
 * @brief The main help message to print to the terminal.
 */
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

/**
 * @brief Convert the passed profiles into the required data structures for
 * processing.
 *
 * @param file The file containing the passed profiles to be used.
 * @param data_names An initialized vector for populating the profile names.
 * @param data_profiles An initialized vector to be populated with the hashed
 * allelic profiles.
 * @param delimiter A character delimiter that can be passed to match the
 * corresponding input file.
 * @param zero_value The zero value used to specify alleles that do not contain
 * a value.
 *
 * @return The header of the file passed.
 *
 * @details
 * This function is responsible for ingestion of the passed input file. It
 * populates the required vectors which contain the data and returns the headers
 * line of the file. The header column is returned so that it can be compared to
 * the header of the second file used by fast matching for verification of a
 * match.
 */
std::string read_profiles(const char *file,
                          std::vector<std::string> &data_names,
                          std::vector<std::vector<uint32_t>> &data_profiles,
                          const char delimiter, const std::string zero_value) {
  std::ifstream fo(file);
  if (!fo.is_open()) {
    std::cerr << "Could not open " << file << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line; // Storage for profile
  std::string header;
  std::getline(fo, header);
  auto columns = std::count(header.begin(), header.end(), delimiter);
  auto line_number = 1; // Starting at 1, as the header value is first.
  while (std::getline(fo, line)) {
    line_number++;
    std::istringstream tokens(line);
    std::string code;
    std::string sample;
    std::getline(tokens, sample, delimiter);
    std::vector<uint32_t> profile(columns);
    size_t idx = 0;

    if (line.empty()) {
      continue;
    }
    auto columns_in_line = std::count(line.begin(), line.end(), delimiter);
    if (columns_in_line != columns) {
      throw std::length_error(
          "Incomplete line in input file: " + std::string(file) +
          " line: " + std::to_string(line_number) + " header has columns " +
          std::to_string(columns) + " only " + std::to_string(columns_in_line) +
          " found.");
    }
    while (std::getline(tokens, code, delimiter)) {
      if (code == zero_value) {
        profile[idx] = MISSING_VALUE;
      } else {
        profile[idx] = static_cast<uint32_t>(std::hash<std::string>{}(code));
      }
      idx++;
    }

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

/**
 *@brief Retrieve the index ranges required for dispatch of each range of
 * samples to a given thread.
 *
 * @param threads The number of threads passed to the program.
 * @param data_size The number of profiles used by the program
 *
 * @return a vector of sample ranges
 *
 * @details
 * This function calls the `sample_ranges` function, however it is a seperate
 * function as it gaurds the logic required for verifying the case when the
 * number of threads passed to program exceeds the number of profiles passed to
 * the program.
 *
 * @usage
 * std::vector<size_t> bins = get_thread_ranges(threads, profiles.size())
 */
std::vector<size_t> get_thread_ranges(size_t threads, size_t data_size) {
  std::vector<size_t> ranges;
  if (threads <= 1 || data_size <= threads) {
    ranges.push_back(0);
    ranges.push_back(data_size);
  } else {
    ranges = sample_ranges(data_size, threads);
  }
  return ranges;
}

// Catch2 provides its own main function and allows for the above functions to
// be included in the test files as a header
#if !defined(TEST) && !defined(PYTHON_BUILD)
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
    std::vector<std::vector<uint32_t>> profiles;
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
    std::vector<std::vector<uint32_t>> query_profiles;
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
    std::cout << "Query\tReference\tDistance" << std::endl;
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
#endif
