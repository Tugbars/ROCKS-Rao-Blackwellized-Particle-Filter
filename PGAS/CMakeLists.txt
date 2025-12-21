#───────────────────────────────────────────────────────────────────────────────
# PGAS Test Bench
#
# Tests and benchmarks for PGAS + PARIS + Lifeboat
# Libraries are defined in parent PGAS/CMakeLists.txt
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)

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