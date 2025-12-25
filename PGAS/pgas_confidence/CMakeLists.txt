#───────────────────────────────────────────────────────────────────────────────
# PGAS Confidence: Confidence Metrics for Adaptive SAEM γ Selection
#
# Computes confidence metrics from PGAS run to inform SAEM blending:
#   - ESS ratio (particle diversity)
#   - Acceptance rate (exploration quality)
#   - Unique fraction (path diversity)
#   - Path divergence (innovation vs reference)
#   - Regime change detection
#
# Outputs adaptive γ ∈ [0.02, 0.15] for normal operation,
# or triggers tier-2 reset (γ = 0.50) on regime change.
#
# Depends on: pgas_mkl (for state access)
#
# Reference: ORACLE_INTEGRATION_PLAN.md
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# PGAS Confidence Library
#───────────────────────────────────────────────────────────────────────────────

add_library(pgas_confidence STATIC
    pgas_confidence.c
)

target_include_directories(pgas_confidence PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/PGAS              # For pgas_mkl.h
)

target_link_libraries(pgas_confidence PUBLIC
    pgas_mkl
)

target_compile_options(pgas_confidence PRIVATE ${COMMON_FLAGS})
target_compile_definitions(pgas_confidence PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(pgas_confidence PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executable
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_pgas_confidence
    test_pgas_confidence.c
)

target_link_libraries(test_pgas_confidence PRIVATE
    pgas_confidence
)

target_compile_options(test_pgas_confidence PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME pgas_confidence_tests COMMAND test_pgas_confidence)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS pgas_confidence
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    pgas_confidence.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  pgas_confidence   - PGAS confidence metrics (9 tests)")
