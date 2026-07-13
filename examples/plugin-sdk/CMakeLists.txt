cmake_minimum_required(VERSION 3.20)
project(vpipe_plugin_example LANGUAGES CXX)

# Point CMake at an installed vpipe prefix, e.g.:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
#   cmake --build build
# Then run it into a session:
#   vpipe --plugin build/example-echo.so --launch-stage example-echo
find_package(vpipe REQUIRED)

# vpipe_add_plugin (shipped by the SDK) builds a dlopen-loadable MODULE
# linked against vpipe::vpipe with C++20. Base plugins need no framework
# flags; add METAL / COREML only for escape-hatch users.
vpipe_add_plugin(example-echo SOURCES example-stage.cc)
