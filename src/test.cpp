#include "main.cpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
    std::string output_test = "t1\tt1\t0.000000\n"
                              "t1\tt2\t66.666672\n"
                              "t1\tt3\t66.666672\n"
                              "t2\tt1\t66.666672\n"
                              "t2\tt2\t0.000000\n"
                              "t2\tt3\t100.000000\n";
    REQUIRE(oss.str() == output_test);
  }
}
