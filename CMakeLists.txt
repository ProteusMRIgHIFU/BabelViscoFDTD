cmake_minimum_required(VERSION 3.11)

set( CMAKE_VERBOSE_MAKEFILE ON )


set (CMAKE_CXX_STANDARD 11)

project(BabelViscoFDTD LANGUAGES C CXX)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(STAGGERED_OPT "Enable platform-specific optimisations" ON)
option(STAGGERED_VERBOSE "Enable verbose output" ON)
option(STAGGERED_SINGLE "Use single precision" ON)
option(STAGGERED_FAST_MATH "Enable unsafe optimisations (non IEEE 754 compliant)" ON)
option(STAGGERED_OMP_SUPPORT "Build with OpenMP support" ON)
option(STAGGERED_CUDA_SUPPORT "Build with CUDA support" ON)
option(STAGGERED_PYTHON_SUPPORT "Build Python C extension module" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_subdirectory(src)
