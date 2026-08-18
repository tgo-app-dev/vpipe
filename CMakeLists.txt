cmake_minimum_required(VERSION 3.25)

# Minimum macOS. Must be set BEFORE project(), which is when the
# compiler is probed and the value baked into every object.
#
# Set to 26.0 DELIBERATELY, and it is not the floor the code imposes.
# The tree compiles and its tests pass at 13.3 (MEASURED: 482 tests
# across 60 suites). What has not been done is running it on a machine
# older than the build host -- availability annotations catch API
# misuse at compile time, not whatever the Metal and CoreML paths do at
# run time on an older OS. Advertising support nobody has exercised is
# a promise this cannot keep, so the claim stays at the version we
# actually test on until someone can try a real older Mac. Lowering it
# is this one line.
#
# When lowering, 13.3 is the true floor, not 13.0, and the .3 is
# load-bearing: std::to_chars for FLOATING-POINT types arrived in macOS
# 13.3 and libc++'s std::format uses it to format doubles, which vpipe
# does everywhere. std::from_chars for floating point is worse still --
# unavailable before macOS 26 -- which is why common/flex-data.cc parses
# doubles with strtod_l on Apple rather than from_chars.
#
# Leaving it UNSET is the one thing that must not happen: clang then
# targets the build host, which shipped binaries stamped minos 26.0
# inside a bundle whose Info.plist advertised 13.0. Those run only on
# the machine that built them, and fail at helper-spawn time with a
# dyld error rather than anything that names the cause.
# FORCE, inside the guard. CMake creates CMAKE_OSX_DEPLOYMENT_TARGET as
# an EMPTY cache entry on Apple, and a plain set(... CACHE ...) never
# overwrites an entry that already exists -- so without FORCE this is a
# no-op in any build tree configured before this line was added, and
# the target silently stays empty. An explicit -D from the user is
# non-empty, so the guard still leaves it alone.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING
        "Minimum macOS version the binaries support" FORCE)
endif()

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
# Stage-provided GUI views (C++ registry + the per-stage JS/CSS embedded
# into libvpipe). After stages/, so a view sits beside the stage it
# presents in the accumulated source list.
add_subdirectory(ui)
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
    # On the compile line so that an FFmpeg major-version change
    # rebuilds every TU, whatever the headers' mtimes claim. See the
    # long note in cmake/find-ffmpeg.cmake: without this, a package
    # upgrade can leave objects built against the previous ABI linked
    # into the same library, which corrupts silently instead of
    # failing. The value is never read by any code.
    target_compile_definitions(vpipe PRIVATE
        VPIPE_FFMPEG_ABI_TAG="${VPIPE_FFMPEG_ABI_TAG}")
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
    # Generated by ui/ (stage-provided view assets embedded into this
    # library); the rule lives in that subdirectory's scope.
    add_dependencies(vpipe vpipe_ui_assets)
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
# Skipped in runtime-compile mode (no metal/metallib toolchain to build it);
# the test then skips vacuously on the empty path.
if(APPLE AND VPIPE_BUILD_APPLE_SILICON AND NOT VPIPE_METAL_RUNTIME_COMPILE)
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
# The interface closure, not just the entry points. Every header below
# is reached TRANSITIVELY from plugin/plugin-context.h or pipeline/stage.h
# -- session-services-intf.h from the plugin context itself, service-req.h
# and the three sink interfaces from Stage -- so omitting them left the
# installed SDK unable to compile even examples/plugin-sdk. A missing
# transitive header is not a smaller SDK, it is a broken one.
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/fs-sandbox-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/log-sink-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/perf-sink-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/runtime-services-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/service-req.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/session-context-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/session-services-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ui-delegate-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ui-port-intf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ui-view-intf.h
    DESTINATION include/interfaces)
# A plugin stage may ship its own GUI view (registry + spec macro); its
# .js/.css are embedded into the plugin's own library the same way.
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/ui-view-registry.h
    DESTINATION include/ui)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/cmake/embed-ui-assets.cmake
    DESTINATION lib/cmake/vpipe)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/edge-reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/oport-buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/resource-plan.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/runtime-context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-registry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage-spec.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/typed-stage.h
    DESTINATION include/pipeline)
# Stage-level SDK for a plugin that adds a MODEL FAMILY: the catalogue
# entry type (so its models are fetchable), the config-source base (so
# its knobs are a stage rather than dead keys on a shared one), and the
# memory helpers every model-holding stage sizes itself with. See
# docs/PLUGINS.md, "Model-specific configuration".
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/stages/model-catalog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stages/model-config-source.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stages/model-memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stages/model-registry.h
    DESTINATION include/stages)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/vae-model-registry.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/video-model-registry.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/hidden-state-encoder.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/weight-set.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/quantize-family-registry.h
        DESTINATION include/generative-models)
    # Reading a Comfy-Org / Comfy-aligned single-file pack (config in the
    # safetensors __metadata__, no config.json). A plugin shipping a model
    # published that way -- which is most new video models -- cannot even
    # detect its own checkpoint without this.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/comfy-checkpoint.h
        DESTINATION include/generative-models/shared)
    # Kernel SELECTION on M5. These carry measurements and, more valuable,
    # recorded NEGATIVE results -- and a plugin that cannot read them does
    # not merely reimplement the picker, it re-runs the experiments and
    # can repeat a mistake this tree already wrote down. That is not
    # hypothetical: an out-of-tree DiT tuned its matmul2d tile with the
    # arms measured SEQUENTIALLY, which is the precise error mma-tile.h
    # records as having turned a 7% regression into an apparent 1.2x win.
    # mma-tile.h is also the rule itself, so an in-tree change to it
    # reaches plugins instead of leaving them on a stale copy.
    #
    # kernel-autotune.h is the keystone: header-only, standard library
    # only, and the dependency of every other file here. It is also the
    # answer to the general problem these encode -- that the fastest
    # kernel is a property of the GPU and the shape, so it should be
    # TIMED rather than predicted from a capability bit.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/kernel-autotune.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/mma-tile.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/mma-splitk.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/i8-gemm.h
        DESTINATION include/generative-models/shared)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/kernel-sets/vae-conv3x3-tune.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/kernel-sets/vae-mid-attn-tune.h
        DESTINATION include/generative-models/shared/kernel-sets)
    # Large-model plumbing. docs/MODEL-MEMORY.md ships as the contract for
    # out-of-tree models and describes block streaming and block residency
    # as things a loader MUST get right; block-residency.h and stream-pin.h
    # are where both actually live, so without them the document describes
    # machinery the reader cannot call. dit-block-progress.h is the reason
    # a high-resolution denoise does not read as a stall -- per-STEP is
    # tens of seconds of silence at 1024px, and every DiT already walks its
    # blocks one at a time.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/block-residency.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/stream-pin.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/shared/dit-block-progress.h
        DESTINATION include/generative-models/shared)
    # weight-set.h's own dependencies. The SDK's header set has to be
    # CLOSED under #include or it does not compile -- a plugin cannot
    # supply a header that lives only in this tree. Verified by walking
    # the closure, not by adding one file per build failure.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/weight-registry.h
        DESTINATION include/generative-models)
    # The model manager. docs/MODEL-MEMORY.md is explicitly the contract
    # for out-of-tree stages too, and two of the things it REQUIRES of a
    # model-holding stage are only reachable through this header:
    # revise_declaration() -- without which a stage that frees a model
    # leaves peers sizing an irreversible streaming decision against
    # weights that are gone -- and park_weights(), the whole middle
    # state of the idle policy. Its include closure (session-member.h,
    # weight-registry.h) is already installed above.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/generative-model-manager.h
        DESTINATION include/generative-models)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/generative-models/llama3/metal-llama-weights.h
        DESTINATION include/generative-models/llama3)
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
    # TensorBeat / TensorBeatPayload. A plugin STAGE that carries tensors
    # -- a conditioner, an encoder, anything feeding generate-video's
    # iport0 -- cannot emit or read a beat without this, so leaving it
    # out made register_stage a promise the SDK could not keep for the
    # numeric stages that most want it.
    # ...with the two headers it includes. The SDK's header set has to be
    # CLOSED under #include, as the weight-set.h block above says; both
    # of these pull in nothing further of their own.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/tensor-beat.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/aligned-allocator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/tensor-storage.h
        DESTINATION include/apple-silicon)
    # CoreML SDK: the model manager ONLY. A CoreML plugin is just a stage
    # that loads a model via session()->coreml_model_manager()->load() and
    # runs it through CoreMLLoadedModel::predict() (neutral CoreMLPredict*
    # structs). All CoreML / Foundation / CoreVideo marshaling lives inside
    # libvpipe, so the vendored coreml-cpp/*.hpp are intentionally NOT
    # installed -- a plugin never needs them.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/apple-silicon/coreml/coreml-model-manager.h
        DESTINATION include/apple-silicon/coreml)
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
