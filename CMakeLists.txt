cmake_minimum_required(VERSION 3.16.3)

if(NOT DEFINED STAGGERED_DEBUG AND DEFINED ENV{STAGGERED_DEBUG})
  set(STAGGERED_DEBUG $ENV{STAGGERED_DEBUG})
endif()
if(NOT DEFINED CMAKE_BUILD_TYPE)
  if(STAGGERED_DEBUG)
      set(CMAKE_BUILD_TYPE "Debug" STRING CACHE "Debug")
    else()
      set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release default")
    endif()
endif()

project(BabelViscoFDTD LANGUAGES C)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_C_STANDARD 11)

option(STAGGERED_OPT "Enable platform-specific optimisations" ON)
option(STAGGERED_VERBOSE "Enable verbose output" ON)
option(STAGGERED_SINGLE "Use single precision" ON)
option(STAGGERED_FAST_MATH "Enable unsafe optimisations (non IEEE 754 compliant)" ON)
option(STAGGERED_OMP_SUPPORT "Build with OpenMP support" ON)
option(STAGGERED_CUDA_SUPPORT "Build with CUDA support" ON)
option(STAGGERED_PYTHON_SUPPORT "Build Python C extension module" ON)

add_subdirectory(src)
