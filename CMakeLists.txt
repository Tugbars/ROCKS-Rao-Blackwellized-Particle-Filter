#───────────────────────────────────────────────────────────────────────────────
# Rao-Blackwellized Particle Filter (RBPF) + PGAS
#
# Standalone build for RBPF-KSC with Kim-Shephard-Chib observation model.
# Optimized for HFT volatility tracking with sub-20μs latency.
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# Options:
#   -DRBPF_USE_DOUBLE=ON     Use double precision (default: OFF for latency)
#   -DRBPF_BUILD_BENCH=ON    Build benchmark executable (default: ON)
#   -DRBPF_BUILD_TESTS=ON    Build scenario tests (default: ON)
#   -DPGAS_BUILD_TESTS=ON    Build PGAS tests (default: ON)
#   -DPGAS_BUILD_BENCH=ON    Build PGAS benchmarks (default: ON)
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(rbpf C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#───────────────────────────────────────────────────────────────────────────────
# MKL Configuration
#───────────────────────────────────────────────────────────────────────────────

set(MKL_INTERFACE lp64)
set(MKL_THREADING intel_thread)  # Intel OpenMP for MSVC
set(MKL_MPI intelmpi)

# Hint for MKL location on Windows
if(WIN32 AND NOT DEFINED ENV{MKLROOT})
    list(APPEND CMAKE_PREFIX_PATH 
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
        "C:/Program Files/Intel/oneAPI/mkl/latest"
    )
endif()

# Find MKL
find_package(MKL CONFIG REQUIRED)

# OpenMP - MKL brings its own Intel OpenMP on Windows
if(MSVC)
    set(OpenMP_FOUND TRUE)
else()
    find_package(OpenMP REQUIRED)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Compiler Flags
#───────────────────────────────────────────────────────────────────────────────

if(MSVC)
    set(COMMON_FLAGS 
        /W4 
        /openmp:llvm     # Use LLVM OpenMP for modern features
        $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    set(COMMON_DEFS _CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
    set(COMMON_FLAGS -Wall -xHost -qopenmp -qopt-report=3 -qopt-report-phase=vec)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(COMMON_FLAGS -Wall -Wextra -Wpedantic -O3 -march=native -fopenmp -ffast-math -funroll-loops)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(COMMON_FLAGS -Wall -Wextra -Wpedantic -O3 -march=native -fopenmp -ffast-math -funroll-loops)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Add Subdirectories
#───────────────────────────────────────────────────────────────────────────────

# Core RBPF library (online filtering)
add_subdirectory(RBPF)

# PGAS + PARIS library (offline smoothing / lifeboat)
add_subdirectory(PGAS)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== RBPF + PGAS Configuration ===")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:         ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "MKL threading:    ${MKL_THREADING}")
if(MSVC)
    message(STATUS "Platform:         Windows (MSVC + Intel OpenMP)")
else()
    message(STATUS "Platform:         Unix-like")
endif()
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  RBPF (online):  rocks library")
message(STATUS "  PGAS (offline): pgas_mkl, paris_mkl, lifeboat")
message(STATUS "")