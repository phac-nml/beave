file(GLOB SOURCES ./*.cpp ./*.hpp)

if(TARGET_GROUP STREQUAL release)
  add_executable(dist-mat ${SOURCES})
  target_include_directories(dist-mat PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(dist-mat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release"
)
elseif(TARGET_GROUP STREQUAL test)
  add_executable(tests test.cpp)
  target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
  target_include_directories(tests PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
  add_custom_command(
    TARGET tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data" "$<TARGET_FILE_DIR:tests>/data"
    COMMENT "Copying ${CMAKE_SOURCE_DIR}/data to $<TARGET_FILE_DIR:tests>"
  )


else()
  add_executable(dist-mat-debug ${SOURCES})
  target_include_directories(dist-mat-debug PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/debug"
)
endif()
