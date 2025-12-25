#───────────────────────────────────────────────────────────────────────────────
# Thompson Sampler: Thompson Sampling for Transition Matrix Handoff
#
# When handing off Π from SAEM to RBPF, balance exploration vs exploitation:
#   - EXPLORE: When row_sum < threshold, sample Π ~ Dirichlet(Q)
#   - EXPLOIT: When row_sum >= threshold, use mean Π = Q / sum(Q)
#
# This prevents over-commitment to uncertain estimates in low-data regimes
# while leveraging high-confidence estimates when available.
#
# No external dependencies - pure C with <math.h>
#
# Reference: ORACLE_INTEGRATION_PLAN.md
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Thompson Sampler Library
#───────────────────────────────────────────────────────────────────────────────

add_library(thompson_sampler STATIC
    thompson_sampler.c
)

target_include_directories(thompson_sampler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(thompson_sampler PRIVATE ${COMMON_FLAGS})
target_compile_definitions(thompson_sampler PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(thompson_sampler PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executable
#───────────────────────────────────────────────────────────────────────────────

add_executable(test_thompson_sampler
    test_thompson_sampler.c
)

target_link_libraries(test_thompson_sampler PRIVATE
    thompson_sampler
)

target_compile_options(test_thompson_sampler PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME thompson_sampler_tests COMMAND test_thompson_sampler)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS thompson_sampler
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    thompson_sampler.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  thompson_sampler  - Thompson sampling for Π handoff")
