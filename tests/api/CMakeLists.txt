set(LOCAL_TEST_SOURCES
  basics.cc
)
set(LOCAL_TEST_HEADERS "")

list(TRANSFORM LOCAL_TEST_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_TEST_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Compiled as a separate OBJECT library so the compiler sees only the
# installed public headers — same view an external consumer would have.
add_library(vpipe_api_tests OBJECT ${LOCAL_TEST_SOURCES})
target_include_directories(vpipe_api_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)

list(APPEND VPIPETEST_SOURCES $<TARGET_OBJECTS:vpipe_api_tests>)
set(VPIPETEST_SOURCES ${VPIPETEST_SOURCES} PARENT_SCOPE)
