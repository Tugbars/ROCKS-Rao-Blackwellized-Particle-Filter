#───────────────────────────────────────────────────────────────────────────────
# PGAS Test Bench: MKL-Optimized PGAS + PARIS + Lifeboat
#
# High-performance particle smoothing for regime-switching stochastic volatility.
#
# Components:
#   - pgas_mkl:    Particle Gibbs with Ancestor Sampling (MKL-accelerated)
#   - paris_mkl:   PARIS backward smoother (MKL-accelerated)  
#   - lifeboat:    Background PGAS+PARIS with lock-free injection into RBPF
#   - circular_buffer: Lock-free SPSC buffer for tick data
#
# Performance (T=200, N=100, K=4):
#   - PGAS sweep:     ~2 ms
#   - PGAS adaptive:  ~6 ms (3 sweeps, 0.95 acceptance)
#   - PARIS smooth:   ~4 ms
#   - Lifeboat total: ~35 ms (background, non-blocking)
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)

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

target_compile_options(pgas_mkl PRIVATE ${COMMON_FLAGS})
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

target_compile_options(paris_mkl PRIVATE ${COMMON_FLAGS})
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

# Need pthreads for background worker thread
find_package(Threads REQUIRED)

#───────────────────────────────────────────────────────────────────────────────
# Combined PGAS Library (convenience target)
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas INTERFACE)
target_link_libraries(pgas INTERFACE
    pgas_mkl
    paris_mkl
    lifeboat
    circular_buffer
)

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

if(PGAS_BUILD_TESTS)

    # PGAS-MKL correctness and benchmark test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_pgas_mkl.c)
        add_executable(test_pgas_mkl
            test_pgas_mkl.c
        )
        target_link_libraries(test_pgas_mkl PRIVATE pgas_mkl)
        target_compile_options(test_pgas_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_pgas_mkl PRIVATE ${COMMON_DEFS})
    endif()

    # PARIS-MKL correctness and benchmark test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_paris_mkl.c)
        add_executable(test_paris_mkl
            test_paris_mkl.c
        )
        target_link_libraries(test_paris_mkl PRIVATE paris_mkl)
        target_compile_options(test_paris_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_paris_mkl PRIVATE ${COMMON_DEFS})
    endif()

    # Lifeboat injection test (lock-free + fast-forward)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_lifeboat.c)
        add_executable(test_lifeboat
            test_lifeboat.c
        )
        target_link_libraries(test_lifeboat PRIVATE lifeboat)
        target_compile_options(test_lifeboat PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_lifeboat PRIVATE ${COMMON_DEFS})
    endif()

    # Circular buffer test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_circular_buffer.c)
        add_executable(test_circular_buffer
            test_circular_buffer.c
        )
        target_link_libraries(test_circular_buffer PRIVATE circular_buffer)
        target_compile_options(test_circular_buffer PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_circular_buffer PRIVATE ${COMMON_DEFS})
    endif()

endif()

#───────────────────────────────────────────────────────────────────────────────
# Benchmark Executables
#───────────────────────────────────────────────────────────────────────────────

if(PGAS_BUILD_BENCH)

    # Full benchmark suite (PGAS + PARIS + Lifeboat)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_all.c)
        add_executable(bench_pgas_all
            benchmark_all.c
        )
        target_link_libraries(bench_pgas_all PRIVATE pgas)
        target_compile_options(bench_pgas_all PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_pgas_all PRIVATE ${COMMON_DEFS})
    endif()

endif()

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
message(STATUS "=== PGAS Test Bench ===")
message(STATUS "Libraries:        pgas_mkl, paris_mkl, lifeboat, circular_buffer")
message(STATUS "Tests enabled:    ${PGAS_BUILD_TESTS}")
message(STATUS "Benchmarks:       ${PGAS_BUILD_BENCH}")
message(STATUS "")
