cmake_minimum_required(VERSION 3.15)

project(beave
  VERSION 0.0.1
  DESCRIPTION "Simple test of distance matrix generation"
  LANGUAGES CXX)
include(FetchContent)
include(CTest)

set(TARGET_GROUP production CACHE STRING "BUILD OPTION")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
    $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
    $ pip install nanobind scikit-build-core[pyproject]
    $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

add_subdirectory(src)
