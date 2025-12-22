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


#───────────────────────────────────────────────────────────────────────────────
# BOCPD Library (required for regime detection)
#
# Architecture: "Pilot Light + Afterburner"
#   - Pilot Light (mutation): Prevents P(regime=k)=0 absorbing states
#   - Afterburner (BOCPD): Event-driven regime switching on vol structure change
#───────────────────────────────────────────────────────────────────────────────

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../RBPF-BOCPD/bocpd.c)
    message(FATAL_ERROR "BOCPD not found at ../RBPF-BOCPD/bocpd.c - BOCPD is required for ROCKS")
endif()

add_library(bocpd STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../RBPF-BOCPD/bocpd.c
)

target_include_directories(bocpd PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../RBPF-BOCPD
)

target_compile_options(bocpd PRIVATE ${COMMON_FLAGS})
target_compile_definitions(bocpd PRIVATE ${COMMON_DEFS})

#───────────────────────────────────────────────────────────────────────────────
# ROCKS Core Library
#───────────────────────────────────────────────────────────────────────────────

set(ROCKS_SOURCES
    # Core Rao-Blackwellized particle filter
    rao-blackwellized/rbpf_ksc.c
    rao-blackwellized/rbpf_ksc_init.c
    rao-blackwellized/rbpf_ksc_output.c
    rao-blackwellized/rbpf_ksc_student_t.c
    
    # Adaptive components
    rao-blackwellized/rbpf_silverman.c   # Silverman bandwidth for regularization
    rao-blackwellized/rbpf_sprt.c        # SPRT regime detection
)

# Add Storvik parameter learning and related modules if enabled
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_SOURCES
        # Storvik online parameter learning
        storvik/rbpf_param_learn.c
        storvik/rbpf_adaptive_forgetting.c
        storvik/rbpf_fixed_lag_smoother.c 

        storvik/rbpf_ksc_param_integration.c      # Core: lifecycle, step, config
        storvik/rbpf_ext_hawkes.c                 # Hawkes + Robust OCSN + Presets
        storvik/rbpf_ext_smoothed_storvik.c       # PARIS smoothed Storvik

         ${CMAKE_CURRENT_SOURCE_DIR}/../PGAS/paris_mkl.c
        
        # Robust OCSN likelihood
        ocsn/rbpf_ocsn_robust.c
        
        # Tuner
        tuner/rbpf_tuner.c
        
        # Preprocessor
        preprocess/rbpf_preprocess.c
        
        # Hawkes intensity
        hawkes/hawkes_intensity.c
        
        # MMPF (Multi-Model Particle Filter) - Institutional Grade
        mmpf/mmpf_core.c
        mmpf/mmpf_api.c
        mmpf/mmpf_entropy.c
        mmpf/mmpf_online_em.c
    )
    
    # MCMC teleportation (optional, requires MKL)
    if(MMPF_ENABLE_MCMC)
        list(APPEND ROCKS_SOURCES
            mmpf/mmpf_mcmc.c
        )
    endif()

    if(RBPF_ENABLE_HDP_BEAM)
    list(APPEND ROCKS_SOURCES
        HDP-Beam/sticky_hdp_beam.c
    )
    endif()
endif()

add_library(rocks STATIC ${ROCKS_SOURCES})

target_include_directories(rocks 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/rao-blackwellized
        ${CMAKE_CURRENT_SOURCE_DIR}/apf
        ${CMAKE_CURRENT_SOURCE_DIR}/storvik
        ${CMAKE_CURRENT_SOURCE_DIR}/ocsn
        ${CMAKE_CURRENT_SOURCE_DIR}/tuner
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
        ${CMAKE_CURRENT_SOURCE_DIR}/hawkes
        ${CMAKE_CURRENT_SOURCE_DIR}/pmmh
        ${CMAKE_CURRENT_SOURCE_DIR}/duration_tracker
        ${CMAKE_CURRENT_SOURCE_DIR}/mmpf
        ${CMAKE_CURRENT_SOURCE_DIR}/HDP-Beam
        ${CMAKE_CURRENT_SOURCE_DIR}/../RBPF-BOCPD
        ${CMAKE_CURRENT_SOURCE_DIR}/../PGAS 
)

target_link_libraries(rocks 
    PUBLIC 
        MKL::MKL
        bocpd
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

# Alias for backward compatibility
add_library(rbpf_ksc ALIAS rocks)

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_TESTS)

    #───────────────────────────────────────────────────────────────────────────
    # MMPF-level tests
    #───────────────────────────────────────────────────────────────────────────

    # MMPF correctness tests
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_correctness.c)
        add_executable(test_mmpf_correctness
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_correctness.c
        )
        target_link_libraries(test_mmpf_correctness PRIVATE rocks)
        target_compile_options(test_mmpf_correctness PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_mmpf_correctness PRIVATE ${COMMON_DEFS})
    endif()

    # MMPF vs Single RBPF comparison test
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_comparison.c)
        add_executable(test_mmpf_comparison
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_comparison.c
        )
        target_link_libraries(test_mmpf_comparison PRIVATE rocks)
        target_compile_options(test_mmpf_comparison PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_mmpf_comparison PRIVATE ${COMMON_DEFS})
    endif()

    # MMPF real data integration test
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/mmpf_real_data_test/test_mmpf_real_data.c)
        add_executable(test_mmpf_real_data
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/mmpf_real_data_test/test_mmpf_real_data.c
        )
        target_link_libraries(test_mmpf_real_data PRIVATE rocks)
        target_compile_options(test_mmpf_real_data PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_mmpf_real_data PRIVATE ${COMMON_DEFS})
    endif()

    # BOCPD standalone test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd.c)
        add_executable(test_bocpd
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd.c
        )
        target_link_libraries(test_bocpd PRIVATE bocpd)
        if(NOT WIN32)
            target_link_libraries(test_bocpd PRIVATE m)
        endif()
        target_compile_options(test_bocpd PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_bocpd PRIVATE ${COMMON_DEFS})
    endif()

    # BOCPD + MMPF integration test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd_mmpf.c)
        add_executable(test_bocpd_mmpf
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd_mmpf.c
        )
        target_link_libraries(test_bocpd_mmpf PRIVATE rocks)
        if(NOT WIN32)
            target_link_libraries(test_bocpd_mmpf PRIVATE m)
        endif()
        target_compile_options(test_bocpd_mmpf PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_bocpd_mmpf PRIVATE ${COMMON_DEFS})
    endif()

    # BOCPD vs Baseline comparison test (targeted scenarios)
    #if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd_comparison.c)
    #    add_executable(test_bocpd_comparison
    #        ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_bocpd_comparison.c
    #    )
    #    target_link_libraries(test_bocpd_comparison PRIVATE rocks)
    #    if(NOT WIN32)
    #        target_link_libraries(test_bocpd_comparison PRIVATE m)
    #   endif()
    #    target_compile_options(test_bocpd_comparison PRIVATE ${COMMON_FLAGS})
    #    target_compile_definitions(test_bocpd_comparison PRIVATE ${COMMON_DEFS})
    #endif()


    #───────────────────────────────────────────────────────────────────────────────
    # HDP-Beam Benchmark
    #───────────────────────────────────────────────────────────────────────────────

    if(RBPF_BUILD_BENCH AND RBPF_ENABLE_HDP_BEAM)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/HDP-Beam/bench_sticky_hdp.c)
        add_executable(bench_sticky_hdp
            ${CMAKE_CURRENT_SOURCE_DIR}/HDP-Beam/bench_sticky_hdp.c
        )
        target_link_libraries(bench_sticky_hdp PRIVATE rocks)
        target_compile_options(bench_sticky_hdp PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_sticky_hdp PRIVATE ${COMMON_DEFS})
        endif()
    endif()

    # Hawkes test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_hawkes.c)
        add_executable(test_hawkes
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_hawkes.c
        )
        target_link_libraries(test_hawkes PRIVATE rocks)
        target_compile_options(test_hawkes PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_hawkes PRIVATE ${COMMON_DEFS})
    endif()

    #───────────────────────────────────────────────────────────────────────────
    # MMPF Parameter Tuners
    #───────────────────────────────────────────────────────────────────────────

    # MMPF Stage 1 Tuner (Static parameters: swim lanes, ν, stickiness)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/mmpf_tuner.c)
        add_executable(mmpf_tuner
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/mmpf_tuner.c
        )
        target_link_libraries(mmpf_tuner PRIVATE rocks)
        target_compile_options(mmpf_tuner PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(mmpf_tuner PRIVATE ${COMMON_DEFS})
    endif()

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


#───────────────────────────────────────────────────────────────────────────────
# Benchmark Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_BENCH)
    # PMMH benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c)
        add_executable(bench_pmmh_mkl
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c
        )
        target_link_libraries(bench_pmmh_mkl PRIVATE rocks)
        target_compile_options(bench_pmmh_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_pmmh_mkl PRIVATE ${COMMON_DEFS})
    endif()

    # PMMH tuning
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c)
        add_executable(tune_pmmh
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c
        )
        target_link_libraries(tune_pmmh PRIVATE rocks)
        target_compile_options(tune_pmmh PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(tune_pmmh PRIVATE ${COMMON_DEFS})
    endif()

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

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS rocks bocpd
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Core headers
set(ROCKS_HEADERS
    mkl_config.h
    rao-blackwellized/rbpf_ksc.h
    rao-blackwellized/rbpf_silverman.h
    rao-blackwellized/rbpf_sprt.h
    rbpf_fisher_rao.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../RBPF-BOCPD/bocpd.h
)

# Storvik and related headers
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_HEADERS
        storvik/rbpf_ksc_param_integration.h
        storvik/rbpf_param_learn.h
        storvik/rbpf_adaptive_forgetting.h
        storvik/p2_quantile.h
        storvik/rbpf_fixed_lag_smoother.h
        tuner/rbpf_tuner.h
        preprocess/rbpf_preprocess.h
        rbpf_dirichlet_transition.h
        hawkes/hawkes_intensity.h
        pmmh/pmmh_mkl.h
        duration_tracker/duration_tracker_fast.h
        mmpf/mmpf_internal.h
        mmpf/mmpf_entropy.h
        mmpf/mmpf_online_em.h
        mmpf/mmpf_mcmc.h
    )
endif()

install(FILES ${ROCKS_HEADERS}
    DESTINATION include/rocks
)

if(MSVC)
    target_compile_options(rocks PRIVATE /openmp:experimental /arch:AVX2)
    target_compile_definitions(rocks PRIVATE HDP_USE_AVX2=1)
endif()