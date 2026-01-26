cmake_minimum_required(VERSION 3.15)

project(dist-mat
  VERSION 0.0.1
  DESCRIPTION "Simple test of distance matrix generation"
  LANGUAGES CXX)
include(FetchContent)
include(CTest)

set(TARGET_GROUP production CACHE STRING "BUILD OPTION")

# Pull boost
macro(pull_boost)
  set(BOOST_ENABLE_PYTHON ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  fetchcontent_declare(Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG  ${BOOST_TAG})
  message(STATUS "Fetching boost ${BOOST_TAG} this may take a bit...")
  fetchcontent_makeavailable(Boost)
  message(STATUS "Retrieved boost ${BOOST_TAG}.")

endmacro()

if(TARGET_GROUP STREQUAL release)

  set(CMAKE_CXX_FLAGS
  "-Wall -Wextra -g -O3 -mfpmath=sse -flto \
  -ffast-math -ftree-vectorize -funroll-loops \
  -fopt-info-vec-optimized -std=c++23")
  #"-Wall -Wextra -g -O3 -march=native -mtune=native -mfpmath=sse
  #-flto -ffast-math
  #-ftree-vectorize -funroll-loops -fopt-info-vec-optimized -std=c++23")
  #"-Wall -Wextra -g -fanalyzer -O3
  #-march=native -mfpmath=sse -flto -fopt-info-vec-all")
  add_subdirectory(src)

elseif(TARGET_GROUP STREQUAL test)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "tests")
  fetchcontent_declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1
  )

  fetchcontent_makeavailable(Catch2)
  add_subdirectory(src)

elseif(TARGET_GROUP STREQUAL python)

  set(BOOST_TAG boost-1.90.0)
  set(BOOST_VERSION 1.90.0)

  find_package(Python3 REQUIRED COMPONENTS Development NumPy)
  find_package(Boost ${BOOST_VERSION} REQUIRED python)


  if(Boost_VERSION_MINOR LESS 90)
    message(STATUS "Boost minor version less than, 90, fetching content.")
    pull_boost()
  elseif(Boost_FOUND)
    message(STATUS "Boost found, skipping fetch content.")
  else()
    message(STATUS "Boost not found, fetching content.")
    pull_boost()
  endif()
  add_subdirectory(src)


else()
  set(CMAKE_CXX_FLAGS "-Wall -O0 -Wextra -g -fanalyzer -std=c++23")
  add_subdirectory(src)
endif()
