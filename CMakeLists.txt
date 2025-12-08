cmake_minimum_required(VERSION 3.15)

project(dist-mat
  VERSION 0.0.1
  DESCRIPTION "Simple test of distance matrix generation"
  LANGUAGES CXX)

set(TARGET_GROUP production CACHE STRING "BUILD OPTION")

if(TARGET_GROUP STREQUAL release)
  set(CMAKE_CXX_FLAGS
  "-Wall -Wextra -g -O3 -march=native -mfpmath=sse -flto -ffast-math -std=c++23")
  #"-Wall -Wextra -g -fanalyzer -O3 -march=native -mfpmath=sse -flto -fopt-info-vec-all")
  add_subdirectory(src)
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -fanalyzer")
  add_subdirectory(src)
endif()
