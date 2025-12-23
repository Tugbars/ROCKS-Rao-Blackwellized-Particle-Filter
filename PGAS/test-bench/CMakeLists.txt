#───────────────────────────────────────────────────────────────────────────────
# PGAS Test Bench
#
# Tests and benchmarks for PGAS libraries
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Test: PGAS-MKL Core
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_pgas_mkl
    test_pgas_mkl.c
)

target_link_libraries(test_pgas_mkl
    PRIVATE
        pgas_mkl
        MKL::MKL
)

target_compile_options(test_pgas_mkl PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_pgas_mkl PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Test: PGAS OCSN Emission
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_pgas_ocsn
    test_pgas_ocsn.c
)

target_link_libraries(test_pgas_ocsn
    PRIVATE
        pgas_mkl
        MKL::MKL
)

target_compile_options(test_pgas_ocsn PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_pgas_ocsn PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Test: PARIS Backward Smoother
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_paris_mkl
    test_paris_mkl.c
)

target_link_libraries(test_paris_mkl
    PRIVATE
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(test_paris_mkl PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_paris_mkl PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Test: Lifeboat Integration
#───────────────────────────────────────────────────────────────────────────────

#add_executable(test_lifeboat
#    test_lifeboat.c
#)

#target_link_libraries(test_lifeboat
#    PRIVATE
#        lifeboat
#        pgas_mkl
#        MKL::MKL
#        Threads::Threads
#)

#target_compile_options(test_lifeboat PRIVATE ${PGAS_COMPILE_FLAGS})
#target_compile_definitions(test_lifeboat PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Validation: Option A - Ground Truth Recovery
#
# Tests PGAS ability to recover KNOWN transition parameters from synthetic data.
# This is the first step in the three-part oracle validation:
#   A. Ground truth recovery (synthetic data with known params)
#   B. Head-to-head vs tuner (same data, different objectives)
#   C. PGAS as warm start for tuner
#───────────────────────────────────────────────────────────────────────────────

add_executable(pgas_validation_a
    pgas_validation_a.c
)

target_link_libraries(pgas_validation_a
    PRIVATE
        pgas_mkl
        MKL::MKL
)

target_compile_options(pgas_validation_a PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_validation_a PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Benchmark: All Components
#───────────────────────────────────────────────────────────────────────────────

add_executable(benchmark_all
    benchmark_all.c
)

target_link_libraries(benchmark_all
    PRIVATE
        pgas_mkl
        paris_mkl
        lifeboat
        MKL::MKL
        Threads::Threads
)

target_compile_options(benchmark_all PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(benchmark_all PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# CTest Integration
#───────────────────────────────────────────────────────────────────────────────

enable_testing()

add_test(NAME pgas_mkl_test COMMAND test_pgas_mkl)
add_test(NAME pgas_ocsn_test COMMAND test_pgas_ocsn)
add_test(NAME paris_mkl_test COMMAND test_paris_mkl)
add_test(NAME lifeboat_test COMMAND test_lifeboat)
add_test(NAME pgas_validation_a_test COMMAND pgas_validation_a)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== PGAS Test Bench ===")
message(STATUS "  test_pgas_mkl      - Core PGAS-MKL tests")
message(STATUS "  test_pgas_ocsn     - OCSN 10-component emission tests")
message(STATUS "  test_paris_mkl     - PARIS backward smoother tests")
message(STATUS "  test_lifeboat      - Lifeboat integration tests")
message(STATUS "  pgas_validation_a  - Option A: Ground truth recovery")
message(STATUS "  benchmark_all      - Full benchmark suite")
message(STATUS "")