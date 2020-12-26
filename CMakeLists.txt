cmake_minimum_required(VERSION 3.11)

set( CMAKE_VERBOSE_MAKEFILE ON )

project(disptools LANGUAGES C)

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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(STAGGERED_PYTHON_C_MODULE_NAME  CACHE STRING "Name of the C extension module file")

# Find OpenMP if required
if(STAGGERED_OMP_SUPPORT)
  #if(APPLE)
  #  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  #      set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
  #      set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument -DUSE_OPENMP" CACHE STRING "" FORCE)
  #      set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
  #      set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
  #      set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
  #      set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
  #  endif()
  #else()
      find_package(OpenMP REQUIRED)
      if(OPENMP_FOUND)
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DUSE_OPENMP")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      endif()
  #endif()
else()
    if(NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

# Enable debug if required
if(STAGGERED_DEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zi")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    endif()
else()
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    endif()
endif()

# Enable non-portable optimisations if required
if(STAGGERED_OPT AND NOT STAGGERED_DEBUG)
    if(NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    endif()
endif()

# Enable unsafe optimisations if required
if(STAGGERED_FAST_MATH AND NOT STAGGERED_DEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
    endif()
endif()


# Use double precision if required
if(STAGGERED_SINGLE)
    add_definitions(-DSINGLE_PREC)
endif()

# Set some compiler flags
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif()

# Add CUDA component if requested
if(STAGGERED_CUDA_SUPPORT)
    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75")

    set(CUDA_SAMPLES_INCLUDE ${CUDA_SAMPLES_LOCATION})

    if(STAGGERED_FAST_MATH AND NOT STAGGERED_DEBUG)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    endif()

    add_definitions(-DCUDA)

endif()

if(STAGGERED_OPENCL_SUPPORT)
    add_definitions(-DOPENCL)
    add_link_options(-Wl,-framework,OpenCL)
endif()

# Add Python component if requested
if(STAGGERED_PYTHON_SUPPORT)
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(PythonInterp 3.6 REQUIRED)

    # Find NumPy headers
    exec_program(${PYTHON_EXECUTABLE}
        ARGS "-c \"import numpy;import os; print(numpy.get_include()+os.sep+'numpy')\""
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        RETURN_VALUE NUMPY_NOT_FOUND
        )
    if(NUMPY_NOT_FOUND)
        message(FATAL_ERROR "NumPy headers not found")
    endif()

    # Find Python headers
    exec_program(${PYTHON_EXECUTABLE}
        ARGS "-c \"import sysconfig; print(sysconfig.get_paths()['include'])\""
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
        RETURN_VALUE PYTHON_INCLUDE_DIRS_NOT_FOUND
        )
    if(PYTHON_INCLUDE_DIRS_NOT_FOUND)
        message(FATAL_ERROR "Python headers not found")
    endif()

    # This goes after, since it uses PythonInterp as a hint
    if(WIN32)
        find_package(PythonLibs 3.6 REQUIRED)
    endif()

    if(STAGGERED_MACOS)
        find_package(PythonLibs 3.6 REQUIRED)
    endif()

    if(STAGGERED_CUDA_SUPPORT)
        set(PYTHON_C_EXTENSION_SRCS
            "FDTDStaggered3D_with_relaxation_python.cu"
        )
    else()
      set(PYTHON_C_EXTENSION_SRCS
          "FDTDStaggered3D_with_relaxation_python.c"
      )
    endif()

    add_library(python_c_extension SHARED ${PYTHON_C_EXTENSION_SRCS})

    set_target_properties(
        python_c_extension
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME ${STAGGERED_PYTHON_C_MODULE_NAME}
            LINKER_LANGUAGE C
    )

    if(WIN32)
        set_target_properties(
            python_c_extension
            PROPERTIES
            SUFFIX ".pyd"
        )
    endif()

    target_include_directories(python_c_extension PUBLIC
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/src/headers
        )

    target_link_libraries(python_c_extension)

    if(STAGGERED_CUDA_SUPPORT)
        target_include_directories(python_c_extension PUBLIC
            ${PROJECT_SOURCE_DIR}/cuda/headers
            ${CUDA_SAMPLES_INCLUDE}
            )
        target_link_libraries(python_c_extension)
    endif()

    # On Windows, it is required to link to the Python libraries
    if(WIN32)
        target_link_libraries(python_c_extension ${PYTHON_LIBRARIES})
    endif()
    if(STAGGERED_MACOS)
        target_link_libraries(python_c_extension ${PYTHON_LIBRARIES})
    endif()
    if(APPLE)
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    endif(APPLE)

endif()
