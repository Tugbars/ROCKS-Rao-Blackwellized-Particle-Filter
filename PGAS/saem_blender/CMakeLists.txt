#───────────────────────────────────────────────────────────────────────────────
# SAEM Blender: Stochastic Approximation EM for Oracle → RBPF Handoff
#
# Safe parameter blending when Oracle (PGAS) produces new transition matrix:
#   - SAEM updates sufficient statistics, not point estimates
#   - Automatic simplex constraint via normalization
#   - Adaptive γ controlled by PGASConfidence
#   - Three-tier reset for regime changes
#   - Floor enforcement for escape hatch
#
# No external dependencies - pure C with <math.h>
#
# Reference: ORACLE_INTEGRATION_PLAN.md v1.6
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# SAEM Blender Library
#───────────────────────────────────────────────────────────────────────────────

add_library(saem_blender STATIC
    saem_blender.c
)

target_include_directories(saem_blender PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(saem_blender PRIVATE ${COMMON_FLAGS})
target_compile_definitions(saem_blender PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(saem_blender PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

# Main SAEM blender tests
add_executable(test_saem_blender
    test_saem_blender.c
)

target_link_libraries(test_saem_blender PRIVATE
    saem_blender
)

target_compile_options(test_saem_blender PRIVATE ${COMMON_FLAGS})

# Three-tier reset tests
add_executable(test_three_tier_reset
    test_three_tier_reset.c
)

target_link_libraries(test_three_tier_reset PRIVATE
    saem_blender
)

target_compile_options(test_three_tier_reset PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME saem_blender_tests COMMAND test_saem_blender)
add_test(NAME three_tier_reset_tests COMMAND test_three_tier_reset)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS saem_blender
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    saem_blender.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  saem_blender      - SAEM parameter blending (2 test suites)")
