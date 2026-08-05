
if(DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER "$ENV{CXX}")
else()
  set(CMAKE_CXX_COMPILER "g++")
endif()

find_package(Python 3.13
    REQUIRED COMPONENTS Interpreter Development.Module
    ${SKBUILD_SABI_COMPONENT})

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    beave_ext
    STABLE_ABI
    NB_STATIC
    ./beave.cpp
)

target_compile_options(beave_ext PRIVATE
    -Wall -Wextra -O3 -mfpmath=sse -fPIC -fpermissive
  -ffast-math -ftree-vectorize -funroll-loops -flto
  -std=c++23 -DPYTHON_BUILD -mavx2)

install(TARGETS beave_ext LIBRARY DESTINATION beave)
install(FILES
  beave/py.typed
  # name must match the name of the file in the imported
  # E.G. function is defined in __init__.py so annotation file in __init__.py
  beave/__init__.pyi
  DESTINATION beave)
