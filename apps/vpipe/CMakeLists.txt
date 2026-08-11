# vpipe -- the command-line entrance to libvpipe. The produced binary is
# named `vpipe`; the CMake target is `vpipe-cli` because the shared library
# target is already named `vpipe` (OUTPUT_NAME restores the binary name).
add_executable(vpipe-cli main.cc)

set_target_properties(vpipe-cli PROPERTIES OUTPUT_NAME vpipe)

target_include_directories(vpipe-cli
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

# Dynamically links the vpipe shared library; the metal-compute / CoreML
# frameworks it needs are linked privately into libvpipe itself.
target_link_libraries(vpipe-cli PRIVATE vpipe)

# libvpipe defines this PRIVATE, so it does not reach here. --gpu-info
# reports on the metal-compute backend and has to know whether that
# backend was built, which is not the same question as "is this a Mac".
if(APPLE AND VPIPE_BUILD_APPLE_SILICON)
    target_compile_definitions(vpipe-cli PRIVATE VPIPE_BUILD_APPLE_SILICON)
    target_include_directories(vpipe-cli PRIVATE ${CMAKE_SOURCE_DIR})
endif()

install(TARGETS vpipe-cli RUNTIME DESTINATION bin)
