set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYTHON_C_EXTENSION_SRCS "FDTDStaggered3D_with_relaxation_python.c")

# ── Compiler flags ────────────────────────────────────────────────────────────
set(_COMMON_OPTS "")
set(_COMMON_DEFS "")

if(STAGGERED_DEBUG)
    if(MSVC)
        list(APPEND _COMMON_OPTS /Zi)
    else()
        list(APPEND _COMMON_OPTS -O0 -g)
    endif()
else()
    if(MSVC)
        list(APPEND _COMMON_OPTS /Ox)
    else()
        list(APPEND _COMMON_OPTS -O3)
    endif()
endif()

if(STAGGERED_OPT AND NOT STAGGERED_DEBUG)
    if(NOT MSVC)
        # -march=native is incompatible with cross-compilation (e.g. building x86_64
        # wheels on an arm64 host via CMAKE_OSX_ARCHITECTURES). Only use it when the
        # target arch matches the host or when no explicit arch override is set.
        if(NOT CMAKE_OSX_ARCHITECTURES
           OR CMAKE_OSX_ARCHITECTURES STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
            list(APPEND _COMMON_OPTS -march=native)
        endif()
    endif()
endif()

if(STAGGERED_FAST_MATH AND NOT STAGGERED_DEBUG)
    if(MSVC)
        list(APPEND _COMMON_OPTS /fp:fast)
    else()
        list(APPEND _COMMON_OPTS -ffast-math)
    endif()
endif()

if(MSVC)
    list(APPEND _COMMON_OPTS /W0)
else()
    list(APPEND _COMMON_OPTS -Wall -Wextra)
endif()

# macOS needs a minimum deployment target to match the Swift dylib
if(APPLE)
    list(APPEND _COMMON_OPTS -mmacosx-version-min=12.0)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
# On macOS, Apple Clang does not bundle OpenMP, so we pre-configure the flags
# that FindOpenMP needs to locate a separately-installed libomp (e.g. from
# Homebrew or conda).  The library name differs by arch.
if(APPLE AND NOT DEFINED OpenMP_C_FLAGS)
    set(OpenMP_C_FLAGS   "-Xclang -fopenmp" CACHE STRING "OpenMP flags for Apple Clang")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp" CACHE STRING "OpenMP flags for Apple Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(OpenMP_C_LIB_NAMES   "omp" CACHE STRING "OpenMP library name")
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP library name")
    else()
        set(OpenMP_C_LIB_NAMES   "iomp5" CACHE STRING "OpenMP library name")
        set(OpenMP_CXX_LIB_NAMES "iomp5" CACHE STRING "OpenMP library name")
    endif()
endif()

set(_OMP_LINK_LIBS "")

if(APPLE OR STAGGERED_OMP_SUPPORT)
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        list(APPEND _COMMON_DEFS USE_OPENMP)
        list(APPEND _OMP_LINK_LIBS OpenMP::OpenMP_C)
    else()
        if(NOT APPLE)
            message(WARNING "OpenMP not found — building without it")
        endif()
        if(NOT MSVC)
            list(APPEND _COMMON_OPTS -Wno-unknown-pragmas)
        endif()
    endif()
else()
    if(NOT MSVC)
        list(APPEND _COMMON_OPTS -Wno-unknown-pragmas)
    endif()
endif()

# ── Helper macro: create one extension target ─────────────────────────────────
macro(add_fdtd_extension _target)
    Python_add_library(${_target} MODULE WITH_SOABI ${PYTHON_C_EXTENSION_SRCS})
    set_target_properties(${_target} PROPERTIES LINKER_LANGUAGE C)

    target_include_directories(${_target} PRIVATE
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_options(${_target} PRIVATE ${_COMMON_OPTS})
    target_compile_definitions(${_target} PRIVATE ${_COMMON_DEFS})

    if(_OMP_LINK_LIBS)
        target_link_libraries(${_target} PRIVATE ${_OMP_LINK_LIBS})
    endif()

    install(TARGETS ${_target} DESTINATION BabelViscoFDTD)
endmacro()

# ── Build single- and double-precision extensions ─────────────────────────────
add_fdtd_extension(_FDTDStaggered3D_with_relaxation_single)
target_compile_definitions(_FDTDStaggered3D_with_relaxation_single PRIVATE SINGLE_PREC)

add_fdtd_extension(_FDTDStaggered3D_with_relaxation_double)
