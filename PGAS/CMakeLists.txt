#───────────────────────────────────────────────────────────────────────────────
# PGAS: Particle Gibbs with Ancestor Sampling
#
# MKL-optimized PGAS + PARIS backward smoother for offline/batch inference.
# Used as "Lifeboat" to rescue online RBPF when VI diverges.
#
# Libraries:
#   - pgas_mkl:        Particle Gibbs with Ancestor Sampling
#   - paris_mkl:       PARIS backward smoother
#   - lifeboat:        Background PGAS+PARIS with lock-free injection
#   - circular_buffer: Lock-free SPSC buffer for tick data
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)

# Need pthreads for lifeboat background worker
find_package(Threads REQUIRED)

#───────────────────────────────────────────────────────────────────────────────
# MSVC-specific: Add /openmp:experimental for SIMD support
#───────────────────────────────────────────────────────────────────────────────

if(MSVC)
    set(PGAS_COMPILE_FLAGS 
        ${COMMON_FLAGS}
        /openmp:experimental
    )
else()
    set(PGAS_COMPILE_FLAGS ${COMMON_FLAGS})
endif()

#───────────────────────────────────────────────────────────────────────────────
# PGAS-MKL Library
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas_mkl STATIC
    pgas_mkl.c
)

target_include_directories(pgas_mkl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pgas_mkl 
    PUBLIC 
        MKL::MKL
)

target_compile_options(pgas_mkl PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_mkl PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# PARIS-MKL Library
#───────────────────────────────────────────────────────────────────────────────

add_library(paris_mkl STATIC
    paris_mkl.c
)

target_include_directories(paris_mkl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(paris_mkl 
    PUBLIC 
        MKL::MKL
)

target_compile_options(paris_mkl PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(paris_mkl PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Circular Buffer Library (Lock-free SPSC)
#───────────────────────────────────────────────────────────────────────────────

add_library(circular_buffer STATIC
    circular_buffer.c
)

target_include_directories(circular_buffer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(circular_buffer PRIVATE ${COMMON_FLAGS})
target_compile_definitions(circular_buffer PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Lifeboat Library (PGAS+PARIS injection system)
#
# Architecture:
#   Main Thread (5µs budget):  RBPF update → atomic ready check → inject
#   Background Thread (~35ms): snapshot → PGAS → PARIS → signal ready
#
# Features:
#   - Lock-free ready check (atomic_load, 1.7ns)
#   - Pre-allocated worker buffers (no malloc in hot path)
#   - xoroshiro128+ RNG (no global mutex)
#   - Fast-forward catch-up for ticks missed during inference
#   - Double-buffered clouds (zero-copy swap)
#───────────────────────────────────────────────────────────────────────────────

add_library(lifeboat STATIC
    lifeboat.c
)

target_include_directories(lifeboat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lifeboat 
    PUBLIC 
        pgas_mkl
        MKL::MKL
    PRIVATE
        Threads::Threads
)

target_compile_options(lifeboat PRIVATE ${COMMON_FLAGS})
target_compile_definitions(lifeboat PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Combined PGAS Library (convenience interface target)
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas INTERFACE)
target_link_libraries(pgas INTERFACE
    pgas_mkl
    paris_mkl
    lifeboat
    circular_buffer
)

#───────────────────────────────────────────────────────────────────────────────
# Test Bench (tests and benchmarks)
#───────────────────────────────────────────────────────────────────────────────

add_subdirectory(test-bench)
add_subdirectory(hawkes_signal)
add_subdirectory(kl_trigger)
add_subdirectory(pgas_confidence)
add_subdirectory(saem_blender)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS pgas_mkl paris_mkl lifeboat circular_buffer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    pgas_mkl.h
    paris_mkl.h
    lifeboat.h
    circular_buffer.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== PGAS Libraries ===")
message(STATUS "  pgas_mkl        - Particle Gibbs with Ancestor Sampling")
message(STATUS "  paris_mkl       - PARIS backward smoother")
message(STATUS "  lifeboat        - Background injection system")
message(STATUS "  circular_buffer - Lock-free SPSC buffer")
message(STATUS "")