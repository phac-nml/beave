cmake_minimum_required(VERSION 3.15)

project(dist-mat
  VERSION 0.0.1
  DESCRIPTION "Simple test of distance matrix generation"
  LANGUAGES CXX)
include(FetchContent)
include(CTest)

set(TARGET_GROUP production CACHE STRING "BUILD OPTION")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

  if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
  else()
    set(DEV_MODULE Development.Module)
  endif()

  find_package(Python 3.13.0 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/nanobind)
  add_subdirectory(src)


else()
  set(CMAKE_CXX_FLAGS "-Wall -O0 -Wextra -g -fanalyzer -std=c++23")
  add_subdirectory(src)
endif()
