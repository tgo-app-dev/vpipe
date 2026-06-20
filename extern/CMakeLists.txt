include(${CMAKE_SOURCE_DIR}/cmake/build-lmdb.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/build-pugixml.cmake)

# metal-cpp is a header-only Apache-2.0 wrapper around the Apple
# Objective-C runtime. We use only its Foundation/* headers (NS::Object,
# NS::String, NS::URL, NS::Error, NS::Dictionary, NS::Array, NS::Number,
# NS::SharedPtr) as the base for our CoreML wrapper. Its own CMakeLists
# builds samples we don't want, so we expose the include directory via
# an INTERFACE target instead of add_subdirectory.
if(APPLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp/Foundation/NSObject.hpp)
    add_library(metal_cpp_foundation INTERFACE)
    target_include_directories(metal_cpp_foundation INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp
    )
endif()

# MLX (https://github.com/ml-explore/mlx) is not built here; the
# metal-compute backend covers the Apple-Silicon compute path. Only the
# MLX Metal kernel headers under gpu-kernels/metal/vendored/mlx/ are used
# (compiled into vpipe's own kernels; see THIRD_PARTY_LICENSES.md).
