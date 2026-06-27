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

install(TARGETS vpipe-cli RUNTIME DESTINATION bin)
