

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(STAGGERED_PYTHON_C_MODULE_NAME  CACHE STRING "Name of the C extension module file")

# Find OpenMP if required
if(STAGGERED_OMP_SUPPORT)
  find_package(OpenMP COMPONENTS C REQUIRED)
  if(OpenMP_C_FOUND)
    add_compile_definitions(USE_OPENMP)
  endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wno-unknown-pragmas>)
endif()

# Enable non-portable optimisations if required
if(STAGGERED_OPT AND NOT STAGGERED_DEBUG)
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-march=native>)
  endif()
endif()

# Enable unsafe optimisations if required
if(STAGGERED_FAST_MATH AND NOT STAGGERED_DEBUG)
  if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/fp:fast>)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffast-math>)
  endif()
endif()


# Use double precision if required
if(STAGGERED_SINGLE)
  add_compile_definitions(SINGLE_PREC)
endif()

# Set some compiler flags
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/W0>)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra>")
endif()

# ------------- Project targets -------------

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

endif()


# --- Python C extension ---
if(STAGGERED_PYTHON_SUPPORT)
  set(Python_FIND_VIRTUALENV FIRST)
  find_package(Python REQUIRED COMPONENTS Development)
  if(Python_VERSION VERSION_LESS 3.8)
    message(FATAL_ERROR "Python version 3.8 or higher is required")
  endif()

  find_package(Python COMPONENTS NumPy REQUIRED)

  # some systems need this extra include dir
  find_path(_numpy_incdir
  NAMES ndarrayobject.h
  HINTS ${Python_NumPy_INCLUDE_DIRS}
  PATH_SUFFIXES numpy
  )
  if(_numpy_incdir)
    list(APPEND Python_NumPy_INCLUDE_DIRS ${_numpy_incdir})
  endif()

  message(STATUS "Python version: ${Python_VERSION}
Numpy Include: ${Python_NumPy_INCLUDE_DIRS}
Python Include: ${Python_INCLUDE_DIRS}
  ")

  set(PYTHON_C_EXTENSION_SRCS FDTDStaggered3D_with_relaxation_python.c)

  add_library(python_c_extension SHARED ${PYTHON_C_EXTENSION_SRCS})

  set_target_properties(python_c_extension
      PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${STAGGERED_PYTHON_C_MODULE_NAME}"
          LINKER_LANGUAGE C
  )
  if(MSVC)
    set_property(TARGET python_c_extension PROPERTY SUFFIX ".pyd")
  endif()

  target_include_directories(python_c_extension PUBLIC
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src/headers
  )

  # On Windows, it is required to link to the Python libraries
  target_link_libraries(python_c_extension PRIVATE
  ${Python_LIBRARIES}
  $<$<BOOL:${OpenMP_C_FOUND}>:OpenMP::OpenMP_C>
  )

endif()
