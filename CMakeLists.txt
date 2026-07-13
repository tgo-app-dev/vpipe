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
add_subdirectory(plugin)
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
            # Root, so a consumer of vpipe::vpipe resolves the SDK's
            # root-relative includes ("pipeline/stage.h", ...) in-tree;
            # the install prefix serves the same layout under include/.
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
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
            "-framework IOKit"
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

# In-repo test plugin: a standalone MODULE .dylib built against libvpipe,
# used by the plugin-loader unit test to exercise the real dlopen ->
# register -> instantiate path. Defined here (not in tests/) because the
# `vpipe` target only exists at root scope after the accumulation subdirs.
# Its path is handed to the test as a compile definition.
add_library(vpipe_test_echo_plugin MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/plugins/echo-plugin/echo-plugin.cc)
set_target_properties(vpipe_test_echo_plugin PROPERTIES PREFIX "")
target_include_directories(vpipe_test_echo_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR})
target_compile_features(vpipe_test_echo_plugin PRIVATE cxx_std_20)
target_link_libraries(vpipe_test_echo_plugin PRIVATE vpipe)
# A two-level-namespace MODULE already errors on undefined symbols by
# default -- so a missing libvpipe symbol fails at plugin-link time rather
# than silently at dlopen. (We must NOT inherit libvpipe's own
# -undefined dynamic_lookup, which is PRIVATE to it and not transitive.)
add_dependencies(vpipe_test vpipe_test_echo_plugin)
if(TARGET vpipe_unit_tests)
    target_compile_definitions(vpipe_unit_tests PRIVATE
        VPIPE_TEST_ECHO_PLUGIN="$<TARGET_FILE:vpipe_test_echo_plugin>")
endif()

# A standalone .metallib compiled offline (NOT embedded into libvpipe),
# mimicking how a plugin ships its own shaders. The plugin-metal-library
# test registers it at runtime via MetalCompute::register_metal_library.
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    set(_probe_src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/plugins/metal-kernel/plugin_probe.metal)
    set(_probe_air ${CMAKE_BINARY_DIR}/plugin_probe.air)
    set(_probe_lib ${CMAKE_BINARY_DIR}/plugin_probe.metallib)
    add_custom_command(
        OUTPUT ${_probe_lib}
        COMMAND xcrun -sdk macosx metal -c ${_probe_src} -o ${_probe_air}
        COMMAND xcrun -sdk macosx metallib ${_probe_air} -o ${_probe_lib}
        DEPENDS ${_probe_src}
        COMMENT "Compiling plugin_probe.metal -> plugin_probe.metallib (test)"
        VERBATIM)
    add_custom_target(vpipe_test_probe_metallib DEPENDS ${_probe_lib})
    add_dependencies(vpipe_test vpipe_test_probe_metallib)
    if(TARGET vpipe_unit_tests)
        target_compile_definitions(vpipe_unit_tests PRIVATE
            VPIPE_TEST_PROBE_METALLIB="${_probe_lib}")
    endif()
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

# ---- plugin SDK: find_package(vpipe) + curated dev headers ----------
# Emit the export set (declared on install(TARGETS) above) and generate a
# package config so an out-of-tree plugin can find_package(vpipe) and link
# vpipe::vpipe. See docs/PLUGINS.md + cmake/vpipePluginHelpers.cmake.
install(EXPORT vpipeTargets
    NAMESPACE vpipe::
    FILE vpipeTargets.cmake
    DESTINATION lib/cmake/vpipe)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vpipeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vpipeConfig.cmake
    INSTALL_DESTINATION lib/cmake/vpipe)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vpipeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)   # SOVERSION 0 == major
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vpipeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vpipeConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vpipePluginHelpers.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vpipe-embed-metallib.cmake
    DESTINATION lib/cmake/vpipe)

# Curated SDK headers, installed under include/ preserving their
# root-relative dirs so `#include "pipeline/stage.h"` resolves from the
# prefix. Lists are the exact include-closures of the plugin entry points
# (stage: typed-stage + runtime-context + job + flex-data + plugin/*).
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/beat-payload-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/flex-data.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/i18n.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/job.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/oport-policy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/session-member.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/vertex.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/vpipe-format.h
    DESTINATION include/common)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/session-context-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ui-delegate-intf.h
    DESTINATION include/interfaces)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/edge-reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/oport-buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/runtime-context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-registry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-spec.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/typed-stage.h
    DESTINATION include/pipeline)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/plugin-abi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/plugin-context.h
    DESTINATION include/plugin)
# Model + Metal SDK (apple-silicon axis only, same gate as the build).
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/context-manager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/loaded-language-model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/model-exec.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/model-exec-registry.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/model-loader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/token-muxer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/tokenizer.h
        DESTINATION include/generative-models)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/buffer-view.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/command-stream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/compute-encoder.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/compute-library.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/event.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/fence.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/metal-compute.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/shared-buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/metal-compute/texture.h
        DESTINATION include/apple-silicon/metal-compute)
    # CoreML SDK: the model manager + the vendored CoreML C++ wrapper. A
    # CoreML plugin is just a stage that loads a model via
    # session()->coreml_model_manager() and decodes tensors to Beats; the
    # vendored coreml-cpp/*.hpp let it touch CML::Model* directly.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/coreml/coreml-model-manager.h
        DESTINATION include/apple-silicon/coreml)
    install(DIRECTORY apple-silicon/coreml/coreml-cpp
        DESTINATION include/apple-silicon/coreml
        FILES_MATCHING PATTERN "*.hpp")
endif()

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
