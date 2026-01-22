#include "main.cpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <vector>

TEST_CASE("Distance Calculations", "[Distance Calculation]") {

  SECTION("Verifying the correct scaled distance is returned.") {
    Output t_out;
    std::vector<uint32_t> t1 = {1, 2, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    t_out.scaled = 0.0;
    REQUIRE(hamming_distance(t1, t2, true, true).scaled == t_out.scaled);
    t1 = {1, 2, 3, 4};
    t2 = {1, 0, 0, 4};
    t_out.scaled = 0.0;
    REQUIRE(hamming_distance(t1, t2, true, false).scaled == t_out.scaled);
    t1 = {2, 2, 3, 4};
    t2 = {1, 0, 0, 4};
    t_out.scaled = 50.0;
    REQUIRE(hamming_distance(t1, t2, true, false).scaled ==
            Catch::Approx(t_out.scaled));
    t1 = {2, 2, 3, 4, 5, 6};
    t2 = {1, 2, 2, 4, 5, 6};
    t_out.scaled = 33.3333;
    REQUIRE(hamming_distance(t1, t2, true, false).scaled ==
            Catch::Approx(t_out.scaled).epsilon(0.0001));
  }

  SECTION("Verifying the correct hamming distance is returned.") {
    Output t_out;
    std::vector<uint32_t> t1 = {1, 2, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    t_out.hamming = 0;
    REQUIRE(hamming_distance(t1, t2, false, true).hamming == t_out.hamming);
    t1 = {1, 2, 3, 4};
    t2 = {1, 0, 0, 4};
    t_out.hamming = 0;
    REQUIRE(hamming_distance(t1, t2, false, false).hamming == t_out.hamming);
    t1 = {2, 2, 3, 4};
    t2 = {1, 0, 0, 4};
    t_out.hamming = 1;
    REQUIRE(hamming_distance(t1, t2, false, false).hamming == t_out.hamming);
    t1 = {2, 2, 3, 4, 5, 6};
    t2 = {1, 2, 2, 4, 5, 6};
    t_out.hamming = 2;
    REQUIRE(hamming_distance(t1, t2, false, false).hamming == t_out.hamming);
  }

  SECTION("Verifying sample ranges function.") {
    CHECK_THROWS(sample_ranges(1, 0));
    CHECK_THROWS(sample_ranges(1, 1));
    REQUIRE(sample_ranges(10, 1) == std::vector<size_t>{0, 10});
    REQUIRE(sample_ranges(100, 10) ==
            std::vector<size_t>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
    REQUIRE(sample_ranges(7, 2) == std::vector<size_t>{0, 3, 6, 7});
  }

  SECTION("Verify get_thread_ranges function.") {
    REQUIRE(get_thread_ranges(0, 100) == std::vector<size_t>{0, 100});
    REQUIRE(get_thread_ranges(1000, 100) == std::vector<size_t>{0, 100});
    REQUIRE(get_thread_ranges(2, 7) == std::vector<size_t>{0, 3, 6, 7});
  }

  // TODO requires more test cases, especially when being executed in different
  // threads
  SECTION(
      "Verify populate_dist_matrix correctly populates the output matrix.") {
    Output v1, v2, v3, v4, v5;
    v1.hamming = 0;
    v2.hamming = 0;
    v3.hamming = 0;
    v4.hamming = 0;
    v5.hamming = 1;
    std::vector<Output> in = {v1, v2, v3, v4};
    std::vector<Output> out = {v1, v5, v5, v1};

    std::vector<uint32_t> t1 = {1, 3, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    std::vector<std::vector<uint32_t>> input = {t1, t2};
    populate_dist_matrix(0, 2, 2, false, false, input, in);
    for (const auto &[e1, e2] : std::views::zip(in, out)) {
      REQUIRE(e1.hamming == e2.hamming);
    }
  }

  SECTION("Verify fasta_match_func outputs for hamming distance.") {
    std::vector<uint32_t> t1 = {1, 2, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    std::vector<uint32_t> t3 = {1, 2, 3, 4};
    std::vector<std::vector<uint32_t>> data = {t1, t2, t3};
    std::vector<std::string> data_names = {"t1", "t2", "t3"};
    auto stdoutBuffer = std::cout.rdbuf(); // save stdout
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());
    fast_match_func(0, 2, false, false, data_names, data);
    std::cout.rdbuf(stdoutBuffer);
    std::string output_test = "t1\tt1\t0\n"
                              "t1\tt2\t0\n"
                              "t1\tt3\t0\n"
                              "t2\tt1\t0\n"
                              "t2\tt2\t0\n"
                              "t2\tt3\t0\n";
    REQUIRE(oss.str() == output_test);
  }

  SECTION("Verify fasta_match_func outputs for scaled distance.") {
    std::vector<uint32_t> t1 = {1, 2, 0, 4};
    std::vector<uint32_t> t2 = {1, 0, 3, 4};
    std::vector<uint32_t> t3 = {0, 2, 3, 4};
    std::vector<std::vector<uint32_t>> data = {t1, t2, t3};
    std::vector<std::string> data_names = {"t1", "t2", "t3"};
    auto stdoutBuffer = std::cout.rdbuf(); // save stdout
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());
    fast_match_func(0, 2, true, true, data_names, data);
    std::cout.rdbuf(stdoutBuffer);
    std::string output_test = "t1\tt1\t0.000000\n"
                              "t1\tt2\t50.000000\n"
                              "t1\tt3\t50.000000\n"
                              "t2\tt1\t50.000000\n"
                              "t2\tt2\t0.000000\n"
                              "t2\tt3\t50.000000\n";
    REQUIRE(oss.str() == output_test);
  }

  SECTION("Verify fasta_match_func outputs for scaled distance missing values "
          "counted.") {
    std::vector<uint32_t> t1 = {1, 2, 1, 1};
    std::vector<uint32_t> t2 = {1, 0, 2, 2};
    std::vector<uint32_t> t3 = {0, 2, 3, 3};
    std::vector<std::vector<uint32_t>> data = {t1, t2, t3};
    std::vector<std::string> data_names = {"t1", "t2", "t3"};
    auto stdoutBuffer = std::cout.rdbuf(); // save stdout
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());
    fast_match_func(0, 2, true, false, data_names, data);
    std::cout.rdbuf(stdoutBuffer);
    // Rounding may be getting weird due to enabling -ffast-math
    std::string output_test = "t1\tt1\t0.000000\n"
                              "t1\tt2\t66.666672\n"
                              "t1\tt3\t66.666672\n"
                              "t2\tt1\t66.666672\n"
                              "t2\tt2\t0.000000\n"
                              "t2\tt3\t100.000000\n";
    REQUIRE(oss.str() == output_test);
  }

  SECTION("Verify read_profiles.") {
    // TODO need to add test verifying lines not fitting the length of the
    // header are deleted

    std::string header = std::string("FILE\tG1\tG2\tG3\tG4\tG5\tG6");
    std::vector<std::string> names;
    std::vector<std::vector<uint32_t>> profiles;
    const char *file = "data/boring.tab";

    std::string output = read_profiles(file, names, profiles, '\t', "0");
    auto columns = std::count(output.begin(), output.end(), '\t');
    auto header_cols = std::count(header.begin(), header.end(), '\t');
    REQUIRE(columns == header_cols);
    CHECK(output == header); // different editors may swap tabs and spaces
    std::vector<std::string> expected_names = {"S1", "S2", "S3",
                                               "S4", "S5", "S6"};
    REQUIRE(expected_names == names);
    std::vector<uint32_t> profile_hashes = {
        4207644323, 1874210838, 3148536061, 1874210838, 4207644323, 1447751201,
        4207644323, 4207644323, 4207644323, 4207644323, 1560837832, 1447751201,
        4207644323, 1874210838, 3148536061, 3337028520, 4207644323, 3148536061,
        4207644323, 3342405555, 1874210838, 3337028520, 4207644323, 3148536061,
        4207644323, 1874210838, 3883143127, 1874210838, 4207644323, 3148536061,
        4207644323, 1874210838, 3883143127, 1874210838, 0,          3148536061,
    };
    uint32_t index = 0;
    for (const auto &profile : profiles) {
      for (const auto &allele : profile) {
        // Using CHECK as the profile hashes may differ due to a differnt
        // compiler
        CHECK(allele == profile_hashes[index]);
        index++;
      }
    }
  }

  SECTION("Verify read_profiles throws errors.") {
    const char *file = "data/boring.mangled.tab";
    std::vector<std::string> names;
    std::vector<std::vector<uint32_t>> profiles;
    CHECK_THROWS(read_profiles(file, names, profiles, '\t', "0"));
  }
}

TEST_CASE("Distance Calculations E2E", "[Matrix Calculations]") {

  SECTION("Test hamming distance logic on file input with known output with 3 "
          "threads.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     */
    const char *file = "data/R1KC1K.tsv";
    char delimiter = '\t';
    std::string zero_value = "0";
    bool count_missing = true;
    bool scaled = false;
    uint8_t threads = 3; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        uint32_t dist = static_cast<uint32_t>(std::abs(sample2 - sample1));
        REQUIRE(output_matrix[(i * profile_names.size()) + f].hamming == dist);
        REQUIRE(output_matrix[(f * profile_names.size()) + i].hamming == dist);
      }
    }
  }

  SECTION("Test scaled distance logic on file input with known output with 2 "
          "threads.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     *
     * Result is always within 4 decimals as it is formatted to 6, with
     * ffast-math enabled
     */
    const char *file = "data/R1KC1K.tsv";
    char delimiter = '\t';
    std::string zero_value = "0";
    bool count_missing = true;
    bool scaled = true;
    uint8_t threads = 2; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        float dist = static_cast<float>(std::abs(sample2 - sample1)) /
                     static_cast<float>(profile_names.size()) * 100.0f;
        INFO("Sample 1: " << sample1);
        INFO("Sample 2: " << sample2);
        INFO("Distance: " << dist);
        INFO("Profiles: " << profile_names.size());
        REQUIRE(output_matrix[(i * profile_names.size()) + f].scaled ==
                Catch::Approx(dist).epsilon(0.0001));
        REQUIRE(output_matrix[(f * profile_names.size()) + i].scaled ==
                Catch::Approx(dist).epsilon(0.0001));
      }
    }
  }

  SECTION("Test scaled hamming logic on file input with known output with 2 "
          "threads and no missing samples and a csv.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     *
     * Result is always within 4 decimals as it is formatted to 6, with
     * ffast-math enabled
     */
    const char *file = "data/R1KC1K.2-zeroes.csv";
    char delimiter = ',';
    std::string zero_value = "0";
    bool count_missing = false;
    bool scaled = false;
    uint8_t threads = 2; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        uint32_t dist = std::abs(sample2 - sample1);
        INFO("Sample 1: " << sample1);
        INFO("Sample 2: " << sample2);
        INFO("Distance: " << dist);
        INFO("Profiles: " << profile_names.size());
        REQUIRE(output_matrix[(i * profile_names.size()) + f].hamming == dist);
        REQUIRE(output_matrix[(f * profile_names.size()) + i].hamming == dist);
      }
    }
  }

  SECTION("Test scaled hamming logic on file input with known output with 2 "
          "being the zero value threads and no missing samples and a csv.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     *
     * Result is always within 4 decimals as it is formatted to 6, with
     * ffast-math enabled
     */
    const char *file = "data/R1KC1K.2-zeroes.csv";
    char delimiter = ',';
    std::string zero_value = "2";
    bool count_missing = true;
    bool scaled = false;
    uint8_t threads = 2; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        uint32_t dist = std::abs(sample2 - sample1);
        INFO("Sample 1: " << sample1);
        INFO("Sample 2: " << sample2);
        INFO("Distance: " << dist);
        INFO("Profiles: " << profile_names.size());
        REQUIRE(output_matrix[(i * profile_names.size()) + f].hamming == dist);
        REQUIRE(output_matrix[(f * profile_names.size()) + i].hamming == dist);
      }
    }
  }

  SECTION("Test scaled hamming logic on file input with known output with 2 "
          "being the zero value threads and missing samples and a csv.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     *
     * Result is always within 4 decimals as it is formatted to 6, with
     * ffast-math enabled
     */
    const char *file = "data/R1KC1K.2-zeroes.csv";
    char delimiter = ',';
    std::string zero_value = "2";
    bool count_missing = false;
    bool scaled = false;
    uint8_t threads = 2; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        uint32_t dist = std::abs(sample2 - sample1);
        INFO("Sample 1: " << sample1);
        INFO("Sample 2: " << sample2);
        INFO("Distance: " << dist);
        INFO("Profiles: " << profile_names.size());
        // No matches as hamming tests for when samples are not equal, and the
        // only non-equal values are 0 values
        REQUIRE(output_matrix[(i * profile_names.size()) + f].hamming == 0);
        REQUIRE(output_matrix[(f * profile_names.size()) + i].hamming == 0);
      }
    }
  }

  SECTION("Test scaled distance logic on file input with known output with 2 "
          "threads, scaled distance, csv input, 2 as the zero value and "
          "missing values not counted.") {
    /*
     * This test is recreates the main logic required for the but it allows for
     * the correctness of outputs to be tested
     *
     * Result is always within 4 decimals as it is formatted to 6, with
     * ffast-math enabled
     */
    const char *file = "data/R1KC1K.tsv";
    char delimiter = ',';
    std::string zero_value = "2";
    bool count_missing = false;
    bool scaled = true;
    uint8_t threads = 2; // use an odd number of threads to mix things up
    std::vector<std::string> profile_names;
    std::vector<std::vector<uint32_t>> profiles;
    profiles.reserve(INITIAL_VEC_SIZE);
    profile_names.reserve(INITIAL_VEC_SIZE);

    // Discarding return value here on purpose
    read_profiles(file, profile_names, profiles, delimiter, zero_value);

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

    // Input sample IDs are integers
    std::vector<int> names;
    names.reserve(profile_names.size());
    for (const auto &name : profile_names) {
      names.emplace_back(std::stoi(name));
    }

    for (size_t i = 0; i < profile_names.size(); i++) {
      int sample1 = names[i];
      for (size_t f = 0; f < profile_names.size(); f++) {
        int sample2 = names[f];
        float dist = static_cast<float>(std::abs(sample2 - sample1)) /
                     static_cast<float>(profile_names.size()) * 100.0f;
        INFO("Sample 1: " << sample1);
        INFO("Sample 2: " << sample2);
        INFO("Distance: " << dist);
        INFO("Profiles: " << profile_names.size());
        REQUIRE(output_matrix[(i * profile_names.size()) + f].scaled == 0.0f);
        REQUIRE(output_matrix[(f * profile_names.size()) + i].scaled == 0.0f);
      }
    }
  }
}

TEST_CASE("Distance Calculations E2E", "[Fast Match Calculations]") {

  SECTION("Fastmatch test on 500 against 500 samples.") {

    const char *input_file = "data/R1KC1K.head.tsv";
    const char *reference_file = "data/R1KC1K.tail.tsv";

    std::vector<std::string> query_names;
    std::vector<std::vector<uint32_t>> query_profiles;
    query_names.reserve(INITIAL_VEC_SIZE);
    query_profiles.reserve(INITIAL_VEC_SIZE);
    char delimiter = '\t';
    std::string zero_value = "0";
    bool count_missing = true;
    bool scaled = false;
    uint8_t threads = 2;

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

    auto stdoutBuffer = std::cout.rdbuf(); // save stdout
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());

    std::vector<std::thread> pool;
    oss << "Query\tReference\tDistance" << std::endl;
    for (size_t i = 0; i < ranges.size() - 1; i++) {
      pool.push_back(std::thread(fast_match_func, ranges[i], ranges[i + 1],
                                 scaled, count_missing, std::cref(query_names),
                                 std::cref(query_profiles)));
    }

    for (std::thread &th : pool) {
      th.join();
    }

    std::cout.rdbuf(stdoutBuffer);

    std::string line;
    std::istringstream data(oss.str());
    std::getline(data, line);
    REQUIRE(line == "Query\tReference\tDistance");
    while (std::getline(data, line)) {
      std::istringstream line_to_parse(line);
      std::string value;
      auto idx = 0;
      std::array<int, 3> values = {0, 0, 0};
      while (std::getline(line_to_parse, value, delimiter)) {
        values[idx] = std::stoi(value);
        idx++;
      }
      REQUIRE(std::abs(values[0] - values[1]) == values[2]);
    }
  }

  SECTION("Fastmatch test on 500 against 500 samples.") {

    const char *input_file = "data/R1KC1K.head.tsv";
    const char *reference_file = "data/R1KC1K.tail.tsv";

    std::vector<std::string> query_names;
    std::vector<std::vector<uint32_t>> query_profiles;
    query_names.reserve(INITIAL_VEC_SIZE);
    query_profiles.reserve(INITIAL_VEC_SIZE);
    char delimiter = '\t';
    std::string zero_value = "0";
    bool count_missing = true;
    bool scaled = true;
    uint8_t threads = 3; // use an odd number of threads to mix things up

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

    auto stdoutBuffer = std::cout.rdbuf(); // save stdout
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());

    std::vector<std::thread> pool;
    oss << "Query\tReference\tDistance" << std::endl;
    for (size_t i = 0; i < ranges.size() - 1; i++) {
      pool.push_back(std::thread(fast_match_func, ranges[i], ranges[i + 1],
                                 scaled, count_missing, std::cref(query_names),
                                 std::cref(query_profiles)));
    }

    for (std::thread &th : pool) {
      th.join();
    }

    std::cout.rdbuf(stdoutBuffer);

    std::string line;
    std::istringstream data(oss.str());
    std::getline(data, line);
    REQUIRE(line == "Query\tReference\tDistance");
    while (std::getline(data, line)) {
      std::istringstream line_to_parse(line);
      std::string value;
      auto idx = 0;
      std::array<int, 2> values = {0, 0};
      float dist = 0.0f;
      while (std::getline(line_to_parse, value, delimiter)) {
        if (idx == 2) {
          dist = std::stof(value);
        } else {
          values[idx] = std::stoi(value);
        }
        idx++;
      }
      float expected_dist =
          (static_cast<float>(std::abs(values[0] - values[1])) /
           static_cast<float>(query_profiles[0].size())) *
          100.0f;
      REQUIRE(expected_dist == Catch::Approx(dist).epsilon(0.0001));
    }
  }
}
