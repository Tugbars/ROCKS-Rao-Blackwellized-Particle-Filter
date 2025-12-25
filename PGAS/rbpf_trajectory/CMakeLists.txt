#───────────────────────────────────────────────────────────────────────────────
# RBPF Trajectory: Trajectory Buffer for RBPF → PGAS Reference Path
#
# PGAS requires a reference trajectory [regimes[T], h[T]] to condition on.
# RBPF doesn't store history by default - this module adds a circular buffer
# to capture the marginal MAP trajectory for Oracle handoff.
#
# Thread-safe single-writer/single-reader with:
#   - Snapshot mode (recommended): atomic copy, zero contention
#   - Seqlock mode: lighter memory, retry on concurrent write
#
# Also provides trajectory tempering for anti-confirmation bias.
#
# No external dependencies - pure C11 with <stdatomic.h>
#
# Reference: ORACLE_INTEGRATION_PLAN.md
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# RBPF Trajectory Library
#───────────────────────────────────────────────────────────────────────────────

add_library(rbpf_trajectory STATIC
    rbpf_trajectory.c
)

target_include_directories(rbpf_trajectory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(rbpf_trajectory PRIVATE ${COMMON_FLAGS})
target_compile_definitions(rbpf_trajectory PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(rbpf_trajectory PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executable
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_rbpf_trajectory
    test_rbpf_trajectory.c
)

target_link_libraries(test_rbpf_trajectory PRIVATE
    rbpf_trajectory
)

target_compile_options(test_rbpf_trajectory PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME rbpf_trajectory_tests COMMAND test_rbpf_trajectory)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS rbpf_trajectory
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    rbpf_trajectory.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  rbpf_trajectory   - Trajectory buffer for Oracle handoff")
