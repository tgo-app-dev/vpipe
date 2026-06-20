# Locate Python and pull in the nanobind submodule. Producing the
# extension module itself happens in python/CMakeLists.txt -- this
# file just makes nanobind_add_module() available.

find_package(Python 3.9
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

# nanobind ships its own CMake config under extern/nanobind/cmake.
# Including its top-level CMakeLists registers the
# nanobind_add_module() helper without building the test suite.
add_subdirectory(
    ${CMAKE_SOURCE_DIR}/extern/nanobind
    ${CMAKE_BINARY_DIR}/extern/nanobind
    EXCLUDE_FROM_ALL
)
