cmake_minimum_required(VERSION 3.25)
project(vpipe VERSION 0.1 LANGUAGES CXX C)

# Default to an optimized build so a plain `cmake -S . -B build` yields peak
# performance. Without this, single-config generators default to an empty
# (== -O0) build type, which silently leaves perf on the table. An explicit
# -DCMAKE_BUILD_TYPE=... always wins; multi-config generators (which carry
# CMAKE_CONFIGURATION_TYPES) pick the config at build time, so skip them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
    message(STATUS "vpipe: CMAKE_BUILD_TYPE not set; defaulting to Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json so clangd/IDEs see the exact per-TU
# flags (the repo uses root-relative includes via include_directories(<root>)
# + generated headers in the build tree, which clangd can't infer otherwise).
# The repo-root .clangd points clangd at the no-MLX build tree's copy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

option(VPIPE_BUILD_PYTHON
    "Build the vpipe Python extension (requires Python dev headers)"
    ON)

# Apple-only feature options (metal-compute / CoreML), declared up here.
if(APPLE)
    # Auto-detect an Apple Silicon (arm64) target so the native
    # metal-compute / CoreML feature set turns on by default. An explicit
    # CMAKE_OSX_ARCHITECTURES wins; otherwise use the target processor
    # (which is the host arch for a native build). Intel Macs default off.
    set(_vpipe_apple_silicon OFF)
    if(CMAKE_OSX_ARCHITECTURES)
        if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
            set(_vpipe_apple_silicon ON)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(_vpipe_apple_silicon ON)
    endif()

    option(VPIPE_BUILD_APPLE_SILICON
        "Build CoreML/metal-compute wrappers + inference stages"
        ${_vpipe_apple_silicon})
    # The metal-compute backend covers the apple-silicon LM / vision /
    # audio path.

    message(STATUS "vpipe: VPIPE_BUILD_APPLE_SILICON="
        "${VPIPE_BUILD_APPLE_SILICON}")
endif()

set(VPIPE_SOURCES
  vpipe-version.cc
)
set(VPIPE_HEADERS "")
set(VPIPETEST_SOURCES "")
set(VPIPETEST_HEADERS "")

add_subdirectory(extern)

if(NOT MSVC)
    add_compile_options(-Wall -Werror)
endif()

# FFmpeg headers (we dlopen the libs at runtime; this is a header-only
# build dependency for the typed function-pointer api structs).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find-ffmpeg.cmake)

# libcurl: REST client stage uses it for HTTP/HTTPS. Always present on
# macOS (SDK ships libcurl.tbd; the dyld shared cache supplies the
# dylib at runtime). On Linux it's a standard package.
find_package(CURL REQUIRED)

add_subdirectory(include)
add_subdirectory(interfaces)
add_subdirectory(common)
add_subdirectory(pipeline)
add_subdirectory(stages)
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    add_subdirectory(apple-silicon)
endif()
# The generative-model (LLM/VLM) subsystem is a top-level tree that builds
# on the apple-silicon metal-compute backend (added above).
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    add_subdirectory(generative-models)
endif()
add_subdirectory(tests)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_target(vpipe_version ALL
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/vpipe-version.h
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DVERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        -DVERSION_MINOR=${PROJECT_VERSION_MINOR}
        -P ${CMAKE_SOURCE_DIR}/cmake/generate-version.cmake
    COMMENT "Generating vpipe-version.h"
    VERBATIM
)

if(VPIPE_SOURCES)
    add_library(vpipe SHARED ${VPIPE_SOURCES} ${VPIPE_HEADERS})
    target_include_directories(vpipe
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}
            ${VPIPE_FFMPEG_INCLUDE_DIRS}
    )
    set_target_properties(vpipe PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_link_libraries(vpipe PRIVATE lmdb pugixml ${CMAKE_DL_LIBS}
                                        CURL::libcurl)
    if(APPLE)
        # Security framework supplies CCRandomGenerateBytes; CoreFoundation
        # is brought in transitively by other Apple bits but link it
        # explicitly so the onvif-discovery stage's KDF/AEAD code works
        # even in builds where APPLE_SILICON is OFF.
        target_link_libraries(vpipe PRIVATE
            "-framework Security"
            "-framework CoreFoundation")
    endif()
    if(APPLE AND VPIPE_BUILD_APPLE_SILICON AND VPIPE_MC_KERNEL_OBJS)
        # Per-kernel OBJECT libraries produced by
        # add_vpipe_metal_kernel(...) calls under
        # apple-silicon/metal-compute/. Linking them as libraries
        # bundles their .o objects into libvpipe.dylib, and CMake
        # chains the upstream xcrun-metal/metallib rules.
        target_link_libraries(vpipe PRIVATE ${VPIPE_MC_KERNEL_OBJS})
    endif()
    if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
        # metal-cpp's NS::ProcessInfo references constants
        # (NSProcessPerformanceProfileDefault/Sustained,
        # NSDeviceCertificationiPhonePerformanceGaming,
        # NSProcessInfoPerformanceProfileDidChangeNotification) that
        # are not exported by every macOS SDK (notably 26.x where
        # these were removed). Use -undefined dynamic_lookup so dyld
        # resolves them at load time -- the metal-cpp init code is
        # weak-import-aware and falls back to default-constructed
        # values when dyld returns null. vpipe itself never reads
        # them.
        target_link_libraries(vpipe PRIVATE
            "-framework CoreML"
            "-framework CoreVideo"
            "-framework Foundation"
            "-framework Metal")
        target_link_options(vpipe PRIVATE
            "LINKER:-undefined,dynamic_lookup")
        target_link_libraries(vpipe PRIVATE metal_cpp_foundation)
        # So shared TUs (Session, the bootstrap-context overrides)
        # can predicate on the apple-silicon backend being built.
        target_compile_definitions(vpipe PRIVATE
            VPIPE_BUILD_APPLE_SILICON)
    endif()
    add_dependencies(vpipe vpipe_version)
else()
    add_library(vpipe INTERFACE)
    target_include_directories(vpipe INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

add_executable(vpipe_test ${VPIPETEST_SOURCES} ${VPIPETEST_HEADERS} ${VPIPE_HEADERS})
target_include_directories(vpipe_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(vpipe_test PRIVATE vpipe lmdb pugixml CURL::libcurl)
if(APPLE)
    target_link_libraries(vpipe_test PRIVATE
        "-framework Security"
        "-framework CoreFoundation")
endif()
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    # Test TUs that include the CoreML wrapper inline objc_msgSend
    # references; resolve them at link time via the same frameworks.
    target_link_libraries(vpipe_test PRIVATE
        "-framework CoreML"
        "-framework CoreVideo"
        "-framework Foundation"
        "-framework Metal"
        metal_cpp_foundation)
endif()

# Applications shipping with libvpipe -- one subdir per app, each
# building its own executable that links the vpipe shared library.
add_subdirectory(apps)

# Python bindings (nanobind). Pulled in after the vpipe target is
# defined so the extension module can link against it.
if(VPIPE_BUILD_PYTHON)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-nanobind.cmake)
    add_subdirectory(python)
endif()

install(TARGETS vpipe
    EXPORT vpipeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/vpipe
    DESTINATION include
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
    DESTINATION share/doc/vpipe
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.md
    DESTINATION share/doc/vpipe
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/extern/lmdb/libraries/liblmdb/LICENSE
    DESTINATION share/doc/vpipe
    RENAME LMDB-LICENSE
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/extern/lmdb/libraries/liblmdb/COPYRIGHT
    DESTINATION share/doc/vpipe
    RENAME LMDB-COPYRIGHT
)
