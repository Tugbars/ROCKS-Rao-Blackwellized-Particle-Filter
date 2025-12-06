#───────────────────────────────────────────────────────────────────────────────
# Rao-Blackwellized Particle Filter (RBPF)
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
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)

project(RBPF
    VERSION 1.0.0
    LANGUAGES C
    DESCRIPTION "Rao-Blackwellized Particle Filter for Stochastic Volatility"
)

#───────────────────────────────────────────────────────────────────────────────
# Build Type
#───────────────────────────────────────────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#───────────────────────────────────────────────────────────────────────────────
# Find MKL
#───────────────────────────────────────────────────────────────────────────────

set(MKL_INTERFACE lp64)
set(MKL_THREADING sequential)  # Single-threaded for low latency

find_package(MKL CONFIG REQUIRED)

if(MKL_FOUND)
    message(STATUS "MKL found: ${MKL_ROOT}")
else()
    message(FATAL_ERROR "Intel MKL not found. Please install oneAPI or set MKL_ROOT.")
endif()

#───────────────────────────────────────────────────────────────────────────────
# Find OpenMP (optional, for non-Windows)
#───────────────────────────────────────────────────────────────────────────────

if(NOT WIN32)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Compiler Flags
#───────────────────────────────────────────────────────────────────────────────

set(COMMON_FLAGS "")
set(COMMON_DEFS "")

if(MSVC)
    # MSVC flags
    set(COMMON_FLAGS
        /O2             # Full optimization
        /arch:AVX2      # AVX2 instructions
        /fp:fast        # Fast floating point
        /GL             # Whole program optimization
        /W3             # Warning level 3
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")
    
    # Suppress some noisy warnings
    list(APPEND COMMON_FLAGS /wd4244 /wd4267 /wd4996)
    
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang flags
    set(COMMON_FLAGS
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -Wall
        -Wextra
        -Wno-unused-parameter
    )
    
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        list(APPEND COMMON_FLAGS -flto)
    endif()
    
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
    # Intel compiler flags
    set(COMMON_FLAGS
        -O3
        -xHost
        -ipo
        -fp-model fast=2
        -Wall
    )
endif()

message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "Flags: ${COMMON_FLAGS}")

#───────────────────────────────────────────────────────────────────────────────
# Add RBPF Subdirectory
#───────────────────────────────────────────────────────────────────────────────

add_subdirectory(RBPF)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════════╗")
message(STATUS "║                    RBPF Build Configuration                   ║")
message(STATUS "╠═══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Version:       ${PROJECT_VERSION}                                       ║")
message(STATUS "║  Build Type:    ${CMAKE_BUILD_TYPE}                                     ║")
message(STATUS "║  Compiler:      ${CMAKE_C_COMPILER_ID}                                        ║")
message(STATUS "║  MKL:           ${MKL_ROOT}")
message(STATUS "╚═══════════════════════════════════════════════════════════════╝")
message(STATUS "")
