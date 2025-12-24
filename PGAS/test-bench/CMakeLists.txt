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

add_executable(test_lifeboat
    test_lifeboat.c
)

target_link_libraries(test_lifeboat
    PRIVATE
        lifeboat
        pgas_mkl
        MKL::MKL
        Threads::Threads
)

target_compile_options(test_lifeboat PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_lifeboat PRIVATE ${COMMON_DEFS})

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
# Validation: Adaptive Kappa
#
# Tests PGAS adaptive κ mechanism that self-tunes stickiness prior.
# Verifies:
#   - κ moves in correct direction when prior mismatches data
#   - Chatter ratio converges toward 1.0
#   - Adaptive outperforms fixed κ when prior is wrong
#───────────────────────────────────────────────────────────────────────────────

add_executable(pgas_validation_adaptive
    pgas_validation_adaptive.c
)

target_link_libraries(pgas_validation_adaptive
    PRIVATE
        pgas_mkl
        MKL::MKL
)

target_compile_options(pgas_validation_adaptive PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_validation_adaptive PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Benchmark: All Components
#───────────────────────────────────────────────────────────────────────────────

#add_executable(benchmark_all
#    benchmark_all.c
#)

#target_link_libraries(benchmark_all
#    PRIVATE
#        pgas_mkl
#        paris_mkl
#        lifeboat
#        MKL::MKL
#        Threads::Threads
#)

#target_compile_options(benchmark_all PRIVATE ${PGAS_COMPILE_FLAGS})
#target_compile_definitions(benchmark_all PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Tool: PGAS Transition Matrix Learner
#
# Learns transition matrix from synthetic data (same as test_mmpf_comparison).
# Outputs copy-pasteable C code for comparison against tuned matrix.
#
# Usage:
#   ./pgas_learn_trans [seed]
#   Default seed = 42 (matches tuner)
#───────────────────────────────────────────────────────────────────────────────

add_executable(pgas_learn_trans
    pgas_learn_trans.c
)

target_link_libraries(pgas_learn_trans
    PRIVATE
        pgas_mkl
        MKL::MKL
)

target_compile_options(pgas_learn_trans PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_learn_trans PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Library: PGAS-PARIS Integration
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas_paris STATIC
    ../pgas_paris.c
)

target_include_directories(pgas_paris
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(pgas_paris
    PUBLIC
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(pgas_paris PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(pgas_paris PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Validation: PGAS vs PGAS-PARIS Comparison
#
# Compares vanilla PGAS against PGAS with PARIS backward smoothing.
# Validates that PGAS-PARIS integration is working correctly.
# Checks:
#   - Both produce valid stochastic matrices
#   - PARIS trajectory diversity > 0% (backward pass working)
#   - Matrices not identical (PARIS actually being used)
#   - Matrices reasonably similar (no fundamental bugs)
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_pgas_vs_paris
    test_pgas_vs_paris.c
)

target_link_libraries(test_pgas_vs_paris
    PRIVATE
        pgas_paris
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(test_pgas_vs_paris PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_pgas_vs_paris PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# CTest Integration
#───────────────────────────────────────────────────────────────────────────────

enable_testing()

add_test(NAME pgas_mkl_test COMMAND test_pgas_mkl)
add_test(NAME pgas_ocsn_test COMMAND test_pgas_ocsn)
add_test(NAME paris_mkl_test COMMAND test_paris_mkl)
add_test(NAME lifeboat_test COMMAND test_lifeboat)
add_test(NAME pgas_validation_a_test COMMAND pgas_validation_a)
add_test(NAME pgas_validation_adaptive_test COMMAND pgas_validation_adaptive)
add_test(NAME pgas_vs_paris_test COMMAND test_pgas_vs_paris)
add_test(NAME regime_learning_test COMMAND test_regime_learning)

#───────────────────────────────────────────────────────────────────────────────
# Test: Full Parameter Learning (μ_vol, σ_h, φ)
#
# Tests PGAS-PARIS learning of all model parameters including shared
# AR dynamics (σ_h via Inv-Gamma, φ via Metropolis-Hastings).
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_full_learning
    test_full_learning.c
)

target_link_libraries(test_full_learning
    PRIVATE
        pgas_paris
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(test_full_learning PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_full_learning PRIVATE ${COMMON_DEFS})

add_test(NAME full_learning_test COMMAND test_full_learning)

#───────────────────────────────────────────────────────────────────────────────
# Benchmark: Transition Matrix Learning (PGAS vs PGAS-PARIS)
#
# Compares speed and accuracy of Π estimation for runtime Lifeboat updates.
# Tests multiple N values (32, 64) with timing and Frobenius distance metrics.
#───────────────────────────────────────────────────────────────────────────────

add_executable(benchmark_trans_learning
    benchmark_trans_learning.c
)

target_link_libraries(benchmark_trans_learning
    PRIVATE
        pgas_paris
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(benchmark_trans_learning PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(benchmark_trans_learning PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Validation: Regime Parameter Learning (PGAS vs PGAS-PARIS)
#
# Side-by-side comparison of μ_vol learning:
#   - Vanilla PGAS: Single reference trajectory
#   - PGAS-PARIS: Rao-Blackwellized ensemble (M trajectories)
#
# Compares:
#   - μ_vol RMSE vs ground truth
#   - Posterior variance (lower = better)
#   - Convergence speed
#   - Π learning quality
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_regime_learning
    test_regime_learning.c
)

target_link_libraries(test_regime_learning
    PRIVATE
        pgas_paris
        pgas_mkl
        paris_mkl
        MKL::MKL
)

target_compile_options(test_regime_learning PRIVATE ${PGAS_COMPILE_FLAGS})
target_compile_definitions(test_regime_learning PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== PGAS Test Bench ===")
message(STATUS "  test_pgas_mkl             - Core PGAS-MKL tests")
message(STATUS "  test_pgas_ocsn            - OCSN 10-component emission tests")
message(STATUS "  test_paris_mkl            - PARIS backward smoother tests")
message(STATUS "  test_lifeboat             - Lifeboat integration tests")
message(STATUS "  pgas_validation_a         - Option A: Ground truth recovery")
message(STATUS "  pgas_validation_adaptive  - Adaptive κ self-tuning")
message(STATUS "  test_pgas_vs_paris        - PGAS vs PGAS-PARIS comparison")
message(STATUS "  test_regime_learning      - μ_vol learning: PGAS vs PGAS-PARIS")
message(STATUS "  test_full_learning        - Full learning: μ_vol + σ_h + φ")
message(STATUS "  benchmark_trans_learning  - Π learning speed/accuracy benchmark")
message(STATUS "  benchmark_all             - Full benchmark suite")
message(STATUS "  pgas_learn_trans          - Learn Π from test_mmpf data")
message(STATUS "")