# Configured in a child process by the top-level CMakeLists.txt before it looks for
# LLVM/Clang itself. A distro's Clang CMake files can be installed without the
# libraries they reference (Debian: the clang-NN package ships ClangTargets.cmake,
# whose static libraries are in libclang-NN-dev), and the vendor file then stops
# configure with a FATAL_ERROR that find_package cannot catch. Failing here, in a
# separate process, lets the console be skipped instead.
cmake_minimum_required(VERSION 3.24)
project(coast_clang_probe C CXX)  # LLVM's CMake package needs Threads, i.e. a language
find_package(LLVM CONFIG REQUIRED)
find_package(Clang CONFIG REQUIRED)
if(NOT TARGET clang-cpp)
    message(FATAL_ERROR "Clang's CMake package defines no clang-cpp target")
endif()
