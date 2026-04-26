cmake_minimum_required(VERSION 3.18)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

project(BabelViscoFDTD LANGUAGES C CXX)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(STAGGERED_OPT          "Enable platform-specific optimisations"                ON)
option(STAGGERED_VERBOSE      "Enable verbose output"                                 ON)
option(STAGGERED_FAST_MATH    "Enable unsafe optimisations (non IEEE 754 compliant)"  ON)
option(STAGGERED_OMP_SUPPORT  "Build with OpenMP support (Linux/Windows)"             ON)
option(STAGGERED_CUDA_SUPPORT "Build with CUDA support"                               ON)
option(STAGGERED_PYTHON_SUPPORT "Build Python C extension module"                     ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# ── Python + NumPy ────────────────────────────────────────────────────────────
find_package(Python 3.6 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ── Generate OpenCL kernel sources and copy indexing headers ──────────────────
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/_prepare_kernels.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _prep_result
)
if(NOT _prep_result EQUAL 0)
    message(FATAL_ERROR "PrepareKernels failed (exit code ${_prep_result})")
endif()

# Install generated OpenCL kernel sources and indexing headers into the wheel.
# These files are gitignored so scikit-build-core's wheel.packages scan won't
# pick them up; explicit install() is the reliable path.
install(FILES
    "${CMAKE_SOURCE_DIR}/BabelViscoFDTD/_gpu_kernel.c"
    "${CMAKE_SOURCE_DIR}/BabelViscoFDTD/_gpu_kernel2D.c"
    "${CMAKE_SOURCE_DIR}/BabelViscoFDTD/_indexing.h"
    "${CMAKE_SOURCE_DIR}/BabelViscoFDTD/_indexing2D.h"
    DESTINATION BabelViscoFDTD
)

# ── macOS: compile Metal shader and build Swift dylib ────────────────────────
if(APPLE)
    set(METAL_SRC_DIR  "${CMAKE_SOURCE_DIR}/src/Metal")
    set(METAL_BUILD_DIR "${CMAKE_BINARY_DIR}/metal_build")

    # Copy Metal package to build tree at configure time so swift build can
    # write its .build/ directory there without touching the source tree.
    file(COPY "${METAL_SRC_DIR}/" DESTINATION "${METAL_BUILD_DIR}")

    set(METAL_AIR  "${METAL_BUILD_DIR}/Sources/BabelMetal/Rayleig.air")
    set(METALLIB   "${METAL_BUILD_DIR}/Sources/BabelMetal/Babel.metallib")
    set(DYLIB      "${METAL_BUILD_DIR}/.build/release/libBabelMetal.dylib")

    # Step 1: .metal → .air
    add_custom_command(
        OUTPUT "${METAL_AIR}"
        COMMAND xcrun -sdk macosx metal
                    -ffast-math
                    -std=macos-metal2.3
                    -mmacosx-version-min=11.0
                    -c Sources/BabelMetal/Babel.metal
                    -o Sources/BabelMetal/Rayleig.air
        WORKING_DIRECTORY "${METAL_BUILD_DIR}"
        DEPENDS "${METAL_BUILD_DIR}/Sources/BabelMetal/Babel.metal"
        COMMENT "Compiling Metal shader → Rayleig.air"
        VERBATIM
    )

    # Step 2: .air → .metallib
    add_custom_command(
        OUTPUT "${METALLIB}"
        COMMAND xcrun -sdk macosx metallib
                    Sources/BabelMetal/Rayleig.air
                    -o Sources/BabelMetal/Babel.metallib
        WORKING_DIRECTORY "${METAL_BUILD_DIR}"
        DEPENDS "${METAL_AIR}"
        COMMENT "Linking Metal library → Babel.metallib"
        VERBATIM
    )

    # Step 3: swift build (produces libBabelMetal.dylib)
    add_custom_command(
        OUTPUT "${DYLIB}"
        COMMAND swift build
                    -Xswiftc -import-objc-header
                    -Xswiftc Sources/BabelMetal/bridge.h
                    -c release
        WORKING_DIRECTORY "${METAL_BUILD_DIR}"
        DEPENDS
            "${METALLIB}"
            "${METAL_BUILD_DIR}/Sources/BabelMetal/BabelMetal.swift"
            "${METAL_BUILD_DIR}/Sources/BabelMetal/bridge.h"
        COMMENT "Building Swift Metal library → libBabelMetal.dylib"
        VERBATIM
    )

    add_custom_target(BabelMetal ALL
        DEPENDS "${DYLIB}" "${METALLIB}"
    )

    install(FILES "${DYLIB}" "${METALLIB}"
        DESTINATION BabelViscoFDTD/tools
    )
endif()

add_subdirectory(src)
