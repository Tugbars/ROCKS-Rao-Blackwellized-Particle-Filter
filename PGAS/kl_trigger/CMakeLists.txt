#───────────────────────────────────────────────────────────────────────────────
# KL Trigger: KL Divergence Trigger for Oracle Dual-Gate
#
# Tracks prediction errors from RBPF and computes KL surprise.
# Works with Hawkes integrator to form the dual-gate trigger:
#   - Hawkes = "Lead" signal (market is getting volatile)
#   - KL = "Truth" signal (filter is actually confused)
#
# Both must fire for Oracle trigger (reduces false positives).
# Panic override bypasses dual-gate for extreme events.
#
# No external dependencies - pure C with <math.h>
#
# Reference: ORACLE_INTEGRATION_PLAN.md
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# KL Trigger Library
#───────────────────────────────────────────────────────────────────────────────

add_library(kl_trigger STATIC
    kl_trigger.c
)

target_include_directories(kl_trigger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(kl_trigger PRIVATE ${COMMON_FLAGS})
target_compile_definitions(kl_trigger PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(kl_trigger PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executable
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_kl_trigger
    test_kl_trigger.c
)

target_link_libraries(test_kl_trigger PRIVATE
    kl_trigger
)

target_compile_options(test_kl_trigger PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME kl_trigger_tests COMMAND test_kl_trigger)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS kl_trigger
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    kl_trigger.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  kl_trigger        - KL divergence trigger (10 tests)")
