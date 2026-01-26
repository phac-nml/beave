/**
 *
 * Important functions used to expose the module outputs as a python interface.
 *
 */

#include "main.hpp"
#include <boost/python.hpp>
#include <boost/python/detail/prefix.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numpy/ndarray.hpp>
#include <boost/python/tuple.hpp>

constexpr size_t INITIAL_VEC_SIZE = 10000;

namespace p = boost::python;
namespace np = boost::python::numpy;

char const *greet() { return "hello, world"; }

void _populate_py_outputs(size_t start, size_t end, size_t pdata_size,
                          const bool scaled, const bool count_missing,
                          const std::vector<std::vector<uint32_t>> profiles,
                          std::vector<float> &output_matrix,
                          size_t matrix_size) {

  for (size_t i = start; i < end; i++) {
    for (size_t f = i; f < pdata_size; f++) {
      Output dist_out =
          hamming_distance(profiles[i], profiles[f], scaled, count_missing);

      // computes upper triangle matrix only
      // output_matrix[(i * pdata_size) + f] = dist_out.scaled;
      output_matrix[(i * matrix_size) - ((i * (i + 1)) / 2) + f] =
          dist_out.scaled;
    }
  }
}

/**
 * Thinking the program will take the file path from the python interpreter
 * parse the columns, perform the computation and pass back the upper triangle
 * matrix for scipy to use
 */
p::object calc_dists(const std::string &text, const char &delimiter,
                     const std::string &zero_value, const uint8_t threads,
                     bool count_missing) {

  std::vector<std::string> profile_names;
  std::vector<std::vector<uint32_t>> profiles;
  profiles.reserve(INITIAL_VEC_SIZE);
  profile_names.reserve(INITIAL_VEC_SIZE);
  const char *file = text.c_str();

  read_profiles(file, profile_names, profiles, delimiter, zero_value);

  std::vector<size_t> ranges = get_thread_ranges(threads, profile_names.size());

  std::vector<std::thread> pool;
  bool scaled = true;

  if (scaled) {
    size_t matrix_size = profile_names.size() * profile_names.size();
    size_t total_upper_elements = (matrix_size * (matrix_size + 1)) / 2;
    std::vector<float> output(total_upper_elements);

    for (size_t i = 0; i < ranges.size() - 1; i++) {
      pool.push_back(std::thread(_populate_py_outputs, ranges[i], ranges[i + 1],
                                 profiles.size(), scaled, count_missing,
                                 std::cref(profiles), std::ref(output),
                                 matrix_size));
    }

    for (std::thread &th : pool) {
      th.join();
    }
    p::object new_owner;

    np::ndarray dout = np::from_data(
        output.data(), np::dtype::get_builtin<float>(),
        p::make_tuple(output.size()), p::make_tuple(output.size()), new_owner);
    return new_owner;
  }
}

BOOST_PYTHON_MODULE(dist_mat) {
  // Scipy neds an upper triangle array, of all real elements e.g. no Nan
  np::initialize();
  p::def("greet", greet);
  p::def("calc_dists", calc_dists);
}
