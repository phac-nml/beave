#ifndef MAIN_HPP
#define MAIN_HPP
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

union Output {
  float scaled;
  uint32_t hamming;
};

Output hamming_distance(const std::vector<uint32_t> &p1,
                        const std::vector<uint32_t> &p2, const bool scaled,
                        const bool count_missing);

std::vector<size_t> get_thread_ranges(size_t threads, size_t data_size);

std::string read_profiles(const char *file,
                          std::vector<std::string> &data_names,
                          std::vector<std::vector<uint32_t>> &data_profiles,
                          const char delimiter, const std::string zero_value);

#endif
