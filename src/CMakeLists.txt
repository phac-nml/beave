file(GLOB SOURCES ./main.cpp ./*.hpp)

if(TARGET_GROUP STREQUAL release)
  add_executable(dist-mat ${SOURCES})
  target_include_directories(dist-mat PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(dist-mat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release"
)
elseif(TARGET_GROUP STREQUAL test)

  add_executable(tests-catch2 test.cpp)
  target_link_libraries(tests-catch2 PRIVATE Catch2::Catch2WithMain)
  target_include_directories(tests-catch2 PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(tests-catch2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
  target_compile_options(tests-catch2 PUBLIC
  -Wall -Wextra -g -O3 -mfpmath=sse
  -ffast-math -ftree-vectorize -funroll-loops
  -std=c++23 -DTEST)

  add_custom_command(
    TARGET tests-catch2
    PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/src/beave/tests/data"
    "$<TARGET_FILE_DIR:tests-catch2>/data"
    COMMENT "Copying ${CMAKE_SOURCE_DIR}/src/beave/tests/data \
    to $<TARGET_FILE_DIR:tests-catch2>"
  )

  add_test(NAME UNIT-TESTS WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests" COMMAND tests-catch2)

  add_executable(tests-exe ${SOURCES})
  target_include_directories(tests-exe PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(tests-exe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )

  target_compile_options(tests-exe PUBLIC
  -Wall -Wextra -g -O3 -mfpmath=sse
  -ffast-math -ftree-vectorize -funroll-loops
  -std=c++23 -flto)


  add_test(NAME E2E-MATRIX-4-THREADS
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  COMMAND sh -c "$<TARGET_FILE:tests-exe> matrix -i data/R1KC1K.tsv -t 4 -c > data/R1KC1K.threaded.test.out")

  add_test(NAME E2E-MATRIX-4-THREADS-COMP
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol data/R1KC1K.out.4.threads.count-missing.tab data/R1KC1K.threaded.test.out)

  add_test(NAME E2E-MATRIX-4-THREADS-SCALED
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  COMMAND sh -c "$<TARGET_FILE:tests-exe> matrix -i data/R1KC1K.tsv -t 4 -sc > data/R1KC1K.threaded.scaled.test.out")

  add_test(NAME E2E-MATRIX-4-THREADS-COMP-SCALED
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol data/R1KC1K.out.4.threads.count-missing.scaled.tab data/R1KC1K.threaded.scaled.test.out)

  add_custom_command(
    TARGET tests-exe
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/src/beave/tests/data"
    "$<TARGET_FILE_DIR:tests-exe>/data"
    COMMENT "Copying ${CMAKE_SOURCE_DIR}/src/beave/tests/data \
    to $<TARGET_FILE_DIR:tests-exe>"
  )

elseif(TARGET_GROUP STREQUAL python)

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
      ./beave.cpp ./main.cpp ./main.hpp
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





else()
  add_executable(dist-mat-debug ${SOURCES})
  target_include_directories(dist-mat-debug PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/debug"
)
endif()
