#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

constexpr size_t MISSING_VALUE = 0;
constexpr size_t MINIMUM_PROFILES = 2;

/**
 * @brief Determine the sample ranges to be calculated based on
 * the number of threads used.
 *
 * @param profiles The number of profiles to be processed
 * @param threads the number of threads used by the program
 *
 * @return A vector of indexes containing the ranges of samples to be
 * partitioned
 *
 * @details
 * The number of threads is handled externally by the program, therefore
 * a value of size 0 should never be passed to the threads argument. There
 * must be at least 2 profiles for any comparison to take place as well.
 *
 * @usage
 * std::vector<size_t> bins = sample_rnages(profiles.size(), threads)
 */
std::vector<size_t> sample_ranges(size_t profiles, size_t threads) {
  if (threads < 1 || profiles < MINIMUM_PROFILES) {
    throw std::invalid_argument("Threads passed must be a positive integer, "
                                "and at least 2 profiles must be passed.");
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
 *@brief Retrieve the index ranges required for partition of each range of
 * samples to a given thread.
 *
 * @param threads The number of threads passed to the program.
 * @param data_size The number of profiles used by the program
 *
 * @return a vector of sample ranges
 *
 * @details
 * This function calls the `sample_ranges` function. However it is a seperate
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

namespace nb = nanobind;

float _hamming_distance(const uint32_t *__restrict__ p1_data,
                        const uint32_t *__restrict__ p2_data, size_t size,
                        const bool scaled, const bool count_missing) {
  uint32_t hamming_distance = 0;
  uint32_t compared_sites = size;

  if (count_missing) {
    for (auto i{size}; i-- > 0;) {
      if (p1_data[i] != p2_data[i]) {
        hamming_distance++;
      }
    }
  } else {
    compared_sites = 0;
    // Hand rolled SIMD instructions for this, but switching to uin32_t allowed
    // the compiler to optimize this code.
    for (auto i{size}; i-- > 0;) {
      const bool valid =
          (p1_data[i] != MISSING_VALUE) & (p2_data[i] != MISSING_VALUE);
      compared_sites += valid;
      hamming_distance += valid & (p1_data[i] != p2_data[i]);
    }
  }

  float distance = static_cast<float>(hamming_distance);
  if (scaled) {
    if (compared_sites) {
      distance = (static_cast<float>(hamming_distance) /
                  static_cast<float>(compared_sites));
    } else {
      distance = 1.0f;
    }
  }

  return distance;
}

/*
 * Interface will take in a numpy array of profiles -1x-1, and return the
 * upper triangle distance matrix only.
 *
 */

using array = nb::ndarray<uint32_t, nb::numpy, nb::shape<-1, -1>, nb::c_contig,
                          nb::device::cpu>;
using array_out = nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig,
                              nb::device::cpu>;

/*
 * Fast matching return value, contains a 3x-1 array. As we compare all of
 * the query sample against themeselves and against all reference samples.
 *
 * Only distances less than a passed thershold are retained in the final
 * output.
 *
 * array positions:
 * position 0 = id of query sample.
 * position 1 = id of reference sample.
 * position 2 = calculated distance.
 *
 */
using array_fast_match = nb::ndarray<float, nb::numpy, nb::shape<3, -1>,
                                     nb::c_contig, nb::device::cpu>;

/*
 * Need to figure out final output storage, will likely need to be an arrray
 * to place nicely with numpy
 **/
void fast_match_function(const array profiles, size_t start, size_t end,
                         const bool scaled, const bool count_missing,
                         float threshold, std::vector<float> &output_data) {
  auto profile_data = profiles.view();
  size_t number_of_loci = profile_data.shape(1);

  for (size_t i = start; i < end; i++) {
    for (size_t f = i + 1;
         f < profiles.shape(0); // shape 0 goes until end of array
         f++) {

      float distance =
          _hamming_distance(&profile_data.data()[i * number_of_loci],
                            &profile_data.data()[f * number_of_loci],
                            number_of_loci, scaled, count_missing);

      if (distance > threshold) {
        continue;
      }

      float query = static_cast<float>(i);
      float reference = static_cast<float>(f);
      output_data.push_back(query);
      output_data.push_back(reference);
      output_data.push_back(distance);
    }
  }
}

void populate_outputs(size_t start, size_t end, size_t pdata_size,
                      const bool scaled, const bool count_missing,
                      const array profiles, float *output_matrix) {

  auto profile_data = profiles.view();
  size_t number_of_loci = profile_data.shape(1);

  for (size_t i = start; i < end; i++) {
    for (size_t f = i + 1; f < profile_data.shape(0); f++) {
      //  Multipling the index by the array length as nd-arrays are stored
      //  linearly
      float distance =
          _hamming_distance(&profile_data.data()[i * number_of_loci],
                            &profile_data.data()[f * number_of_loci],
                            number_of_loci, scaled, count_missing);

      // Compute upper triangle position for the 1D array
      size_t upper_triangle_pos =
          ((pdata_size * (pdata_size - 1)) / 2) -
          ((pdata_size - i) * (pdata_size - i - 1) / 2) + f - i - 1;

      output_matrix[upper_triangle_pos] = distance;
    }
  }
}

array_out calculate_distances(array np_in, size_t threads, bool scaled,
                              bool count_missing) {

  // Store the number of profiles required
  size_t number_profiles = np_in.shape(0);
  // Determine the thread ranges to be used
  std::vector<size_t> ranges = get_thread_ranges(threads, number_profiles);
  std::vector<std::thread> pool;

  // Calculate the total space needed to contain the final number of outputs
  // in the upper triangle.
  size_t total_upper_elements = (number_profiles * (number_profiles - 1)) / 2;

  float *output = new float[total_upper_elements];

  for (size_t i = 0; i < ranges.size() - 1; i++) {
    pool.push_back(std::thread(populate_outputs, ranges[i], ranges[i + 1],
                               number_profiles, scaled, count_missing,
                               std::cref(np_in), std::ref(output)));
  }
  for (std::thread &th : pool) {
    th.join();
  }

  nb::capsule owner(output, [](void *p) noexcept { delete[] (float *)p; });
  return array_out(output, {total_upper_elements}, owner);
}

/*
 * @brief Driver function for fast_match function
 *
 * @param np_in The input numpy array from python containing the profiles used
 * for calculation.
 * @param threads The number of cores to use for the calculation.
 * @param scaled If True return the hamming distance as a proportion of
 * comparisons made.
 * @param count_missing If True count missing values as data.
 * @param query_length The number of query profiles with the array.
 **/
array_fast_match fast_match(array np_in, size_t threads, bool scaled,
                            bool count_missing, size_t query_length,
                            float threshold) {

  std::vector<size_t> thread_ranges = get_thread_ranges(threads, query_length);

  std::vector<std::thread> pool;

  std::vector<std::vector<float>> results;

  for (size_t i = 0; i < thread_ranges.size() - 1; i++) {
    std::vector<float> vec;
    vec.reserve(10000);
    results.push_back(std::move(vec));
    /*
     * emplace_back could be used here, but reserve can only be called
     * after initialization. This means we have to create the vector first.
     * */
  }

  for (size_t i = 0; i < thread_ranges.size() - 1; i++) {
    pool.push_back(std::thread(fast_match_function, std::cref(np_in),
                               thread_ranges[i], thread_ranges[i + 1], scaled,
                               count_missing, threshold, std::ref(results[i])));
  }

  for (std::thread &th : pool) {
    th.join();
  }

  // Copying data from vector to final array as returning a pointer to the
  // vector data results in a segmentation fault as the destructor is called
  // on the vector when this function exits. However python still has the
  // reference to the data.

  size_t recorded_results = results[0].size();
  // Get capacity of each filled vector
  if (results.size() > 1) {
    recorded_results = std::ranges::fold_left(
        results.begin() + 1, results.end(), recorded_results,
        [](size_t acc, const std::vector<float> &x) { return acc + x.size(); });
  }

  float *output = new float[recorded_results];
  size_t output_diff = 0;
  for (size_t i = 0; i < results.size(); i++) {
    size_t bytes_copy = results[i].size() * sizeof(float);
    std::memcpy(&output[output_diff], results[i].data(), bytes_copy);
    output_diff += results[i].size();
  }

  nb::capsule owner(output, [](void *p) noexcept { delete[] (float *)p; });
  constexpr size_t records_per_row = 3;
  size_t rows =
      recorded_results / records_per_row; // Should be at most 3 values
  return array_fast_match(output, {rows, records_per_row}, owner);
}

NB_MODULE(beave_ext, m) {

  m.doc() = "Fast distance matrix computation exploiting simd intrinsics and "
            "C++ parallelism."; // module docstring
  m.def("calc_dists", &calculate_distances,
        "Calculate hamming distance between all sets of profiles.");
  m.def("fast_match", &fast_match,
        "Calculate pairwise distances of query profiles against all other "
        "input profiles.");
  m.attr("missing_value") = MISSING_VALUE;
}
