#include "main.cpp"
#include <catch2/catch_config.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

// Example function
// TEST_CASE("Hello test world", "[hello world]") { REQUIRE(test1() == 1); }

TEST_CASE("Distance Calculations", "[Distance Calculation]") {

  std::mt19937 gen(42);

  SECTION("Verifying the correct scaled distance is returned.") {
    Output t_out;
    std::vector<uint32_t> t1 = {1, 2, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    t_out.scaled = 0.0;
    REQUIRE(hamming_distance(t1, t2, true, true).scaled == t_out.scaled);
  }

  SECTION("Verifying the correct hamming distance is returned.") {
    Output t_out;
    std::vector<uint32_t> t1 = {1, 2, 3, 4};
    std::vector<uint32_t> t2 = {1, 2, 3, 4};
    t_out.hamming = 0;
    REQUIRE(hamming_distance(t1, t2, false, true).hamming == t_out.hamming);
  }
}
