#───────────────────────────────────────────────────────────────────────────────
# Oracle Bridge: Complete Oracle Integration Pipeline
#
# Integrates all Oracle sub-components into a unified pipeline:
#
#   ┌────────────────────────────────────────────────────────────────────┐
#   │  RBPF Thread                       │  Oracle Thread                │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  [Hawkes + KL] ──trigger?────────► │  [Scout] pre-validate         │
#   │                                    │     │                         │
#   │  [RBPFTrajectory] ──snapshot─────► │  [PGAS] run + confidence      │
#   │                                    │     │                         │
#   │  [RBPF Π update] ◄──Thompson───── │  [SAEM] blend with γ(conf)    │
#   └────────────────────────────────────────────────────────────────────┘
#
# Dependencies (all Oracle sub-components):
#   - hawkes_integrator  (trigger signal #1)
#   - kl_trigger         (trigger signal #2 - dual gate)
#   - pgas_mkl           (Oracle sampler)
#   - paris_mkl          (Scout sweep - optional)
#   - pgas_confidence    (confidence metrics → adaptive γ)
#   - saem_blender       (safe parameter blending)
#   - thompson_sampler   (explore/exploit handoff)
#
# Reference: ORACLE_INTEGRATION_PLAN.md
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Oracle Bridge Library
#───────────────────────────────────────────────────────────────────────────────

add_library(oracle_bridge STATIC
    oracle_bridge.c
)

target_include_directories(oracle_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Sub-component includes
    ${CMAKE_CURRENT_SOURCE_DIR}/../hawkes_signal
    ${CMAKE_CURRENT_SOURCE_DIR}/../kl_trigger
    ${CMAKE_CURRENT_SOURCE_DIR}/../pgas_confidence
    ${CMAKE_CURRENT_SOURCE_DIR}/../saem_blender
    ${CMAKE_CURRENT_SOURCE_DIR}/../thompson_sampler
    ${CMAKE_CURRENT_SOURCE_DIR}/../rbpf_trajectory
    # Main PGAS includes (pgas_mkl.h, paris_mkl.h)
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link all Oracle sub-components
target_link_libraries(oracle_bridge PUBLIC
    # Trigger components
    hawkes_integrator
    kl_trigger
    # PGAS (from main PGAS folder)
    pgas_mkl
    # Confidence and blending
    pgas_confidence
    saem_blender
    thompson_sampler
    # Trajectory buffer (optional, for snapshot mode)
    rbpf_trajectory
)

# Optional: PARIS scout sweep support
# Set ORACLE_BRIDGE_USE_PARIS=1 to enable scout pre-validation
option(ORACLE_BRIDGE_USE_PARIS "Enable PARIS scout sweep support" OFF)
if(ORACLE_BRIDGE_USE_PARIS)
    target_link_libraries(oracle_bridge PUBLIC paris_mkl)
    target_compile_definitions(oracle_bridge PUBLIC ORACLE_BRIDGE_USE_PARIS=1)
    message(STATUS "    → PARIS scout enabled")
else()
    target_compile_definitions(oracle_bridge PUBLIC ORACLE_BRIDGE_USE_PARIS=0)
    message(STATUS "    → PARIS scout disabled (set ORACLE_BRIDGE_USE_PARIS=ON to enable)")
endif()

target_compile_options(oracle_bridge PRIVATE ${COMMON_FLAGS})
target_compile_definitions(oracle_bridge PRIVATE ${COMMON_DEFS})

# Link math library on Unix
if(UNIX)
    target_link_libraries(oracle_bridge PUBLIC m)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

# Basic integration test
add_executable(test_oracle_integration
    test_oracle_integration.c
)

target_link_libraries(test_oracle_integration PRIVATE
    oracle_bridge
)

target_compile_options(test_oracle_integration PRIVATE ${COMMON_FLAGS})

# Full integration test (realistic scenarios)
add_executable(test_oracle_full_integration
    test_oracle_full_integration.c
)

target_link_libraries(test_oracle_full_integration PRIVATE
    oracle_bridge
)

target_compile_options(test_oracle_full_integration PRIVATE ${COMMON_FLAGS})

# Register with CTest
add_test(NAME oracle_integration COMMAND test_oracle_integration)
add_test(NAME oracle_full_integration COMMAND test_oracle_full_integration)

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS oracle_bridge
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    oracle_bridge.h
    DESTINATION include/pgas
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "  oracle_bridge     - Full Oracle integration pipeline")
message(STATUS "                      Components: hawkes + kl + pgas + paris + confidence + saem + thompson")