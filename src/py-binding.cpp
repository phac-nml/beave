/**
 *
 * Important functions used to expose the module outputs as a python interface.
 *
 */

#include "main.cpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;
char const *greet() { return "hello, world"; }

/**
 * Thinking the program will take the file path from the python interpreter
 * parse the columns, perform the computation and pass back the upper triangle
 * matrix for scipy to use
 */

BOOST_PYTHON_MODULE(dist_mat) {
  // Scipy neds an upper triangle array, of all real elements e.g. no Nan

  p::def("greet", greet);
}
