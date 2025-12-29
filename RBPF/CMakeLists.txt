#───────────────────────────────────────────────────────────────────────────────
# ROCKS: Rao-blackwellized Online Conjugate KSC-Storvik
# Robust real-time stochastic volatility tracking for high-frequency trading
#───────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)

#───────────────────────────────────────────────────────────────────────────────
# Options
#───────────────────────────────────────────────────────────────────────────────

option(RBPF_USE_DOUBLE "Use double precision (slower but more accurate)" OFF)
option(RBPF_BUILD_BENCH "Build benchmark executable" ON)
option(RBPF_BUILD_TESTS "Build test executables" ON)
option(RBPF_ENABLE_STORVIK "Enable Sleeping Storvik parameter learning" ON)
option(MMPF_ENABLE_MCMC "Enable MCMC particle teleportation" ON)
option(RBPF_ENABLE_HDP_BEAM "Enable Sticky HDP-HMM Beam Sampling" ON)
option(RBPF_ENABLE_ONLINE_VI "Enable Online VI transition learning" OFF)

#───────────────────────────────────────────────────────────────────────────────
# ROCKS Core Library
#───────────────────────────────────────────────────────────────────────────────

set(ROCKS_SOURCES
    # Core Rao-Blackwellized particle filter
    rao-blackwellized/rbpf_ksc.c
    rao-blackwellized/rbpf_ksc_init.c
    rao-blackwellized/rbpf_ksc_output.c
    rao-blackwellized/rbpf_ksc_student_t.c
    rao-blackwellized/rbpf_kl_tempering.c
    
    # Adaptive components
    rao-blackwellized/rbpf_silverman.c   # Silverman bandwidth for regularization
    rao-blackwellized/rbpf_sprt.c        # SPRT regime detection
    rao-blackwellized/rbpf_mh_jitter.c   # MH jittering (Stone #2 - informed exploration)
)

# Add Storvik parameter learning and related modules if enabled
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_SOURCES
        # Storvik online parameter learning
        storvik/rbpf_param_learn.c
        storvik/rbpf_adaptive_forgetting.c
        storvik/rbpf_fixed_lag_smoother.c 

        storvik/rbpf_ksc_param_integration.c      # Core: lifecycle, step, config
        storvik/rbpf_ext_smoothed_storvik.c       # PARIS smoothed Storvik

        ${CMAKE_CURRENT_SOURCE_DIR}/../PGAS/paris_mkl.c
        
        # Robust OCSN likelihood
        ocsn/rbpf_ocsn_robust.c
        
        # Preprocessor
        preprocess/rbpf_preprocess.c

    )
endif()

add_library(rocks STATIC ${ROCKS_SOURCES})

target_include_directories(rocks 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/rao-blackwellized
        ${CMAKE_CURRENT_SOURCE_DIR}/storvik
        ${CMAKE_CURRENT_SOURCE_DIR}/ocsn
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
        ${CMAKE_CURRENT_SOURCE_DIR}/pmmh
        ${CMAKE_CURRENT_SOURCE_DIR}/duration_tracker
        ${CMAKE_CURRENT_SOURCE_DIR}/../PGAS
)

target_link_libraries(rocks 
    PUBLIC 
        MKL::MKL
)

target_compile_options(rocks PRIVATE ${COMMON_FLAGS})
target_compile_definitions(rocks PRIVATE ${COMMON_DEFS})

if(RBPF_USE_DOUBLE)
    target_compile_definitions(rocks PUBLIC RBPF_USE_DOUBLE=1)
endif()

if(RBPF_ENABLE_STORVIK)
    target_compile_definitions(rocks PUBLIC RBPF_ENABLE_STORVIK=1)
endif()

if(MMPF_ENABLE_MCMC)
    target_compile_definitions(rocks PUBLIC MMPF_ENABLE_MCMC=1)
endif()

if(RBPF_ENABLE_ONLINE_VI)
    target_compile_definitions(rocks PUBLIC RBPF_ENABLE_ONLINE_VI=1)
endif()

# Alias for backward compatibility
add_library(rbpf_ksc ALIAS rocks)

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

# Single RBPF Test (Storvik + KL Tempering + Smoothed Storvik + OCSN)
      # Single RBPF Test (Storvik + KL Tempering + Smoothed Storvik + OCSN)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_single_rbpf.c)
        add_executable(test_single_rbpf
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_single_rbpf.c
        )
        target_link_libraries(test_single_rbpf PRIVATE rocks)
        target_compile_options(test_single_rbpf PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_single_rbpf PRIVATE ${COMMON_DEFS})
    endif()


if(RBPF_BUILD_TESTS)

     # RBPF Parameter Tuner (Multi-threaded)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_tuner_mp.c)
        add_executable(rbpf_tuner_mp
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_tuner_mp.c
        )
        target_link_libraries(rbpf_tuner_mp PRIVATE rocks)
        target_compile_options(rbpf_tuner_mp PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(rbpf_tuner_mp PRIVATE ${COMMON_DEFS})
    endif()

    # Example usage
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c)
        add_executable(example_usage
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c
        )
        target_link_libraries(example_usage PRIVATE rocks)
        target_compile_options(example_usage PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(example_usage PRIVATE ${COMMON_DEFS})
    endif()
endif()

# RBPF Visualization Export
if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_viz_export.c)
    add_executable(rbpf_viz_export
        ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_viz_export.c
    )
    target_link_libraries(rbpf_viz_export PRIVATE rocks)
    target_compile_options(rbpf_viz_export PRIVATE ${COMMON_FLAGS})
    target_compile_definitions(rbpf_viz_export PRIVATE ${COMMON_DEFS})
endif()

add_executable(test_pgas_injection test/test_pgas_injection.c)
target_link_libraries(test_pgas_injection PRIVATE
    rocks
    pgas_oracle
    pgas_sliding
    MKL::MKL
)
target_compile_options(test_pgas_injection PRIVATE ${COMMON_FLAGS})
target_compile_definitions(test_pgas_injection PRIVATE ${COMMON_DEFS})
#───────────────────────────────────────────────────────────────────────────────
# Benchmark Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_BENCH)

    # Student-t MKL benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_student_t.c)
        add_executable(bench_student_t_mkl
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_student_t.c
        )
        target_link_libraries(bench_student_t_mkl PRIVATE rocks)
        target_compile_options(bench_student_t_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_student_t_mkl PRIVATE ${COMMON_DEFS})
    endif()
endif()

add_executable(test_stone1_diagnostic
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_stone1_diagnostic.c
)
target_link_libraries(test_stone1_diagnostic PRIVATE rocks)

# ═══════════════════════════════════════════════════════════════════════════════
# Test: Hawkes Timing Test (MLE calibration + timing validation)
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(test_hawkes_timing
    test/test_hawkes_timing.c
)

target_include_directories(test_hawkes_timing PRIVATE
    ${CMAKE_SOURCE_DIR}/PGAS/hawkes_signal      # hawkes_integrator.h
    ${CMAKE_SOURCE_DIR}/Hawkes_MLE              # smc2_hawkes_mle.h
)

target_link_libraries(test_hawkes_timing PRIVATE
    hawkes_integrator    # Real-time Hawkes (from PGAS/hawkes_signal)
    hawkes_mle           # MLE calibration (from Hawkes_MLE)
)

target_compile_options(test_hawkes_timing PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME hawkes_timing COMMAND test_hawkes_timing)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

# Core headers
set(ROCKS_HEADERS
    mkl_config.h
    rao-blackwellized/rbpf_ksc.h
    rao-blackwellized/rbpf_silverman.h
    rao-blackwellized/rbpf_sprt.h
    rao-blackwellized/rbpf_mh_jitter.h
    rao-blackwellized/rbpf_kl_tempering.h
    rbpf_fisher_rao.h
)

# Storvik and related headers
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_HEADERS
        storvik/rbpf_ksc_param_integration.h
        storvik/rbpf_param_learn.h
        storvik/rbpf_adaptive_forgetting.h
        storvik/p2_quantile.h
        storvik/rbpf_fixed_lag_smoother.h
        preprocess/rbpf_preprocess.h
    )
endif()

install(FILES ${ROCKS_HEADERS}
    DESTINATION include/rocks
)

if(MSVC)
    target_compile_options(rocks PRIVATE /openmp:experimental /arch:AVX2)
    target_compile_definitions(rocks PRIVATE HDP_USE_AVX2=1)
endif()