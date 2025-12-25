#───────────────────────────────────────────────────────────────────────────────
# Hawkes Signal: Integrated Intensity Trigger for Oracle Activation
#
# Hawkes process-based anomaly detection with:
#   - Integrated intensity (not spike detection)
#   - Hysteresis thresholds (high/low water marks)
#   - Refractory period (prevent rapid re-triggers)
#   - Variance-based self-calibrating thresholds
#
# MKL is optional - provides vectorized exp() for kernel computation.
# Without MKL, falls back to scalar expf() which is still fast enough.
#
# Literature: Hawkes (1971), adapted for regime change detection
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Hawkes Integrator Library
#───────────────────────────────────────────────────────────────────────────────

add_library(hawkes_integrator STATIC
    hawkes_integrator.c
)

target_include_directories(hawkes_integrator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(hawkes_integrator PRIVATE ${COMMON_FLAGS})
target_compile_definitions(hawkes_integrator PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(hawkes_integrator PUBLIC m)
endif()

# Optional MKL acceleration
if(TARGET MKL::MKL)
    target_link_libraries(hawkes_integrator PUBLIC MKL::MKL)
    target_compile_definitions(hawkes_integrator PRIVATE USE_MKL=1)
    message(STATUS "  hawkes_integrator - with MKL acceleration")
else()
    message(STATUS "  hawkes_integrator - scalar fallback (no MKL)")
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executable
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_hawkes_scenarios
    test_hawkes_scenarios.c
)

target_link_libraries(test_hawkes_scenarios PRIVATE
    hawkes_integrator
)

target_compile_options(test_hawkes_scenarios PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME hawkes_scenarios COMMAND test_hawkes_scenarios)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS hawkes_integrator
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    hawkes_integrator.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  hawkes_integrator - Integrated intensity trigger (6 tests)")