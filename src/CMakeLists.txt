file(GLOB SOURCES ./*.cpp ./*.hpp)

if(TARGET_GROUP STREQUAL release)
  add_executable(dist-mat ${SOURCES})
  target_include_directories(dist-mat PUBLIC ${CMAKE_CURRENT_LIST_DIR})
else()

  add_executable(dist-mat-debug ${SOURCES})
  target_include_directories(dist-mat-debug PUBLIC ${CMAKE_CURRENT_LIST_DIR})
endif()
