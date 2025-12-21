#───────────────────────────────────────────────────────────────────────────────
# PGAS Test Bench
#
# Tests and benchmarks for PGAS + PARIS + Lifeboat
# Libraries are defined in parent PGAS/CMakeLists.txt
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

# PGAS-MKL correctness and benchmark test
add_executable(test_pgas_mkl
    test_pgas_mkl.c
)
target_link_libraries(test_pgas_mkl PRIVATE pgas_mkl)
target_compile_options(test_pgas_mkl PRIVATE ${COMMON_FLAGS})
target_compile_definitions(test_pgas_mkl PRIVATE ${COMMON_DEFS})

# PARIS-MKL correctness and benchmark test
add_executable(test_paris_mkl
    test_paris_mkl.c
)
target_link_libraries(test_paris_mkl PRIVATE paris_mkl)
target_compile_options(test_paris_mkl PRIVATE ${COMMON_FLAGS})
target_compile_definitions(test_paris_mkl PRIVATE ${COMMON_DEFS})

# Lifeboat injection test (lock-free + fast-forward)
add_executable(test_lifeboat
    test_lifeboat.c
)
target_link_libraries(test_lifeboat PRIVATE lifeboat)
target_compile_options(test_lifeboat PRIVATE ${COMMON_FLAGS})
target_compile_definitions(test_lifeboat PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Benchmark Executables
#───────────────────────────────────────────────────────────────────────────────

# Full benchmark suite (PGAS + PARIS + Lifeboat)
#add_executable(bench_pgas_all
#    benchmark_all.c
#)
#target_link_libraries(bench_pgas_all PRIVATE pgas)
#target_compile_options(bench_pgas_all PRIVATE ${COMMON_FLAGS})
#target_compile_definitions(bench_pgas_all PRIVATE ${COMMON_DEFS})