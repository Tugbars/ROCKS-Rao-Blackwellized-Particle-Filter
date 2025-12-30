# CMakeLists.txt for sr_detector module
# Part of Crisis_Detection subsystem
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j
#
# Options:
#   -DUSE_MKL=ON/OFF       Force MKL on/off (auto-detected by default)
#   -DSIMD_ARCH=native     SIMD architecture (native, avx512, avx2, sse4)
#
cmake_minimum_required(VERSION 3.16)
project(sr_detector VERSION 1.0.0 LANGUAGES C)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
option(USE_MKL "Use Intel MKL for vectorized math" ON)
option(BUILD_TESTS "Build test executable" ON)
option(BUILD_BENCH "Build benchmark executable" ON)
set(SIMD_ARCH "native" CACHE STRING "SIMD architecture: native, avx512, avx2, sse4, none")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER FLAGS
# ═══════════════════════════════════════════════════════════════════════════════
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Base flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# SIMD flags
if(SIMD_ARCH STREQUAL "native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
elseif(SIMD_ARCH STREQUAL "avx512")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=skylake-avx512")
elseif(SIMD_ARCH STREQUAL "avx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=haswell")
elseif(SIMD_ARCH STREQUAL "sse4")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.2")
endif()

# FMA (Fused Multiply-Add)
include(CheckCCompilerFlag)
check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_FMA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# FIND MKL
# ═══════════════════════════════════════════════════════════════════════════════
set(MKL_FOUND FALSE)

if(USE_MKL)
    # Try to find MKL via environment
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT $ENV{MKLROOT})
    elseif(DEFINED ENV{ONEAPI_ROOT})
        set(MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest")
    elseif(EXISTS "/opt/intel/oneapi/mkl/latest")
        set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
    elseif(EXISTS "/opt/intel/mkl")
        set(MKL_ROOT "/opt/intel/mkl")
    endif()

    if(MKL_ROOT)
        # Check if MKL headers exist
        if(EXISTS "${MKL_ROOT}/include/mkl.h")
            set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
            
            # Determine library path
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
                if(NOT EXISTS "${MKL_LIB_DIR}")
                    set(MKL_LIB_DIR "${MKL_ROOT}/lib")
                endif()
            else()
                set(MKL_LIB_DIR "${MKL_ROOT}/lib/ia32")
            endif()

            # Check for libraries
            find_library(MKL_CORE_LIB mkl_core PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
            find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
            find_library(MKL_SEQUENTIAL_LIB mkl_sequential PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)

            if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB)
                set(MKL_FOUND TRUE)
                set(MKL_LIBRARIES 
                    ${MKL_INTEL_LP64_LIB}
                    ${MKL_SEQUENTIAL_LIB}
                    ${MKL_CORE_LIB}
                )
                message(STATUS "Found MKL: ${MKL_ROOT}")
            endif()
        endif()
    endif()

    if(NOT MKL_FOUND)
        message(STATUS "MKL not found, using SIMD fallback")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY: sr_detector
# ═══════════════════════════════════════════════════════════════════════════════
add_library(sr_detector STATIC
    sr_detector.c
    sr_detector.h
)

target_include_directories(sr_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link math library
target_link_libraries(sr_detector PUBLIC m)

# MKL integration
if(MKL_FOUND)
    target_compile_definitions(sr_detector PUBLIC __INTEL_MKL__)
    target_include_directories(sr_detector PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(sr_detector PUBLIC ${MKL_LIBRARIES})
    target_link_libraries(sr_detector PUBLIC pthread dl)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTABLE: test_sr_detector
# ═══════════════════════════════════════════════════════════════════════════════
if(BUILD_TESTS)
    add_executable(test_sr_detector test_sr_detector.c)
    target_link_libraries(test_sr_detector PRIVATE sr_detector)
    
    # Enable CTest
    enable_testing()
    add_test(NAME sr_detector_tests COMMAND test_sr_detector)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTABLE: bench_sr_detector
# ═══════════════════════════════════════════════════════════════════════════════
if(BUILD_BENCH)
    add_executable(bench_sr_detector bench_sr_detector.c)
    target_link_libraries(bench_sr_detector PRIVATE sr_detector)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  sr_detector Configuration Summary")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD arch:      ${SIMD_ARCH}")
message(STATUS "  MKL:            ${MKL_FOUND}")
message(STATUS "  Build tests:    ${BUILD_TESTS}")
message(STATUS "  Build bench:    ${BUILD_BENCH}")
message(STATUS "  C flags:        ${CMAKE_C_FLAGS}")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")
