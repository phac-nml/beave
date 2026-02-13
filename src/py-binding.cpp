/**
 *
 * Important functions used to expose the module outputs as a python interface.
 *
 */

#include "main.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <thread>
#include <vector>

namespace nb = nanobind;

float _hamming_distance(const uint32_t *__restrict__ p1_data,
                        const uint32_t *__restrict__ p2_data, size_t size,
                        const bool scaled, const bool count_missing) {
  float dist_out;
  uint32_t dist = 0;
  uint32_t compared_sites = size;

  if (count_missing) {
    for (size_t i = 0; i < size; i++) {
      if (p1_data[i] != p2_data[i]) {
        dist++;
      }
    }
  } else {
    compared_sites = 0;
    // Hand rolled SIMD instructions for this, but switching to uin32_t allowed
    // the compiler to optimize this code.
    for (size_t i = 0; i < size; i++) {
      const bool valid =
          (p1_data[i] != MISSING_VALUE) & (p2_data[i] != MISSING_VALUE);
      compared_sites += valid;
      dist += valid & (p1_data[i] != p2_data[i]);
    }
  }

  dist_out = dist;
  if (scaled) {
    dist_out = (static_cast<float>(dist) / static_cast<float>(compared_sites)) *
               100.0f;
  }

  return dist_out;
}

/*
 * Interface will take in a numpy array of profiles 2x2, and return the upper
 * triangle distance matrix only.
 */

using array = nb::ndarray<uint32_t, nb::numpy, nb::c_contig, nb::device::cpu>;
using array_out = nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu>;

void populate_outputs(size_t start, size_t end, size_t pdata_size,
                      const bool scaled, const bool count_missing,
                      const array profiles, float *output_matrix) {

  auto profile_data = profiles.data();

  // get the length of the profiles used
  size_t profiles_used = profiles.shape(1);
  for (size_t i = start; i < end; i++) {
    for (size_t f = i; f < pdata_size; f++) {
      float dist_out = _hamming_distance(&profile_data[i], &profile_data[f],
                                         profiles_used, scaled, count_missing);

      // Compute upper triangle position
      size_t upper_triangle_pos =
          (i * profiles_used) - ((i * (i - 1)) / 2) + (f - i);
      output_matrix[upper_triangle_pos] = dist_out;
    }
  }
}

array_out calculate_distances(array np_in, size_t threads, bool scaled,
                              bool count_missing) {

  // Store the number of profiles required
  size_t number_profiles = np_in.shape(0);
  std::vector<std::vector<uint32_t>> profiles(number_profiles);

  // Determine the thread ranges to be used
  std::vector<size_t> ranges = get_thread_ranges(threads, number_profiles);
  std::vector<std::thread> pool;

  // Calculate the total space needed to contain the final number of outputs in
  // the upper triangle
  size_t total_upper_elements = (number_profiles * (number_profiles + 1)) / 2;

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

// define Python module, expose py_cube function as "cube" to python
NB_MODULE(dist_mat, m) {

  m.doc() = "Fast distance matrix computation exploiting simd intrinsics and "
            "C++ parallelism."; // module docstring
  m.def("calc_dists", &calculate_distances);
  m.attr("missing_value") = MISSING_VALUE;
}

// Example code below
// C/C++ implementation of the function to be wrapped
// void c_cube(const double *v_in, double *v_out, size_t n_elem) {
//  for (size_t i = 0; i < n_elem; ++i) {
//    v_out[i] = v_in[i] * v_in[i] * v_in[i];
//  }
//}
//
//
//// wrapper function, accepting a NumPy array as input and returning a NumPy
//// array
// array py_cube(array np_in) {
//   // create output buffer
//   double *out_buffer = new double[np_in.size()];
//
//   // call C/C++ function with proper arguments
//   c_cube(np_in.data(), out_buffer, np_in.size());
//
//   // Delete 'data' when the 'owner' capsule expires
//   nb::capsule owner(out_buffer, [](void *p) noexcept { delete[] (double *)p;
//   });
//
//   return array(out_buffer, np_in.ndim(), (const size_t *)np_in.shape_ptr(),
//                owner);
// }
