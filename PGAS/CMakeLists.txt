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
# PGAS Oracle - Background thread for sliding window PGAS
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas_oracle STATIC
    pgas_oracle.c
)

target_include_directories(pgas_oracle PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pgas_oracle 
    PUBLIC 
        pgas_sliding
        MKL::MKL
        Threads::Threads
)

target_compile_options(pgas_oracle PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_oracle PRIVATE ${COMMON_DEFS})

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

add_library(pgas_sliding STATIC
    pgas_sliding.c
)
target_include_directories(pgas_sliding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pgas_sliding PUBLIC pgas_mkl MKL::MKL)
target_compile_options(pgas_sliding PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_sliding PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# PGAS + RBPF Coexistence Test
#───────────────────────────────────────────────────────────────────────────────

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_pgas_rbpf_coexist.c)
    add_executable(test_pgas_rbpf_coexist
        test_pgas_rbpf_coexist.c
    )
    target_link_libraries(test_pgas_rbpf_coexist PRIVATE 
        pgas_oracle
        pgas_sliding
        pgas_mkl
        rocks
        MKL::MKL
        Threads::Threads
    )
    target_include_directories(test_pgas_rbpf_coexist PRIVATE
        ${CMAKE_SOURCE_DIR}/RBPF
        ${CMAKE_SOURCE_DIR}/RBPF/rao-blackwellized
        ${CMAKE_SOURCE_DIR}/RBPF/storvik
        ${CMAKE_SOURCE_DIR}/RBPF/ocsn
    )
    target_compile_options(test_pgas_rbpf_coexist PRIVATE ${PGAS_COMPILE_FLAGS})
    target_compile_definitions(test_pgas_rbpf_coexist PRIVATE ${COMMON_DEFS})
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Bench (tests and benchmarks)
#───────────────────────────────────────────────────────────────────────────────

add_subdirectory(test-bench)
add_subdirectory(hawkes_signal)
add_subdirectory(kl_trigger)
add_subdirectory(pgas_confidence)
add_subdirectory(saem_blender)
add_subdirectory(thompson_sampler)
add_subdirectory(rbpf_trajectory)

# Top-level integration (depends on all above)
add_subdirectory(oracle_bridge)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS pgas_mkl paris_mkl
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    pgas_mkl.h
    paris_mkl.h
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