

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(STAGGERED_PYTHON_C_MODULE_NAME  CACHE STRING "Name of the C extension module file")

# Find OpenMP if required
if(STAGGERED_OMP_SUPPORT)
      find_package(OpenMP REQUIRED)
      if(OPENMP_FOUND)
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DUSE_OPENMP")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      endif()
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

    
    set(PYTHON_C_EXTENSION_SRCS
          "FDTDStaggered3D_with_relaxation_python.c"
      )


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

    # On Windows, it is required to link to the Python libraries

    target_link_libraries(python_c_extension ${PYTHON_LIBRARIES})

    
    if(APPLE)
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    endif(APPLE)

endif()
