#───────────────────────────────────────────────────────────────────────────────
# RBPF Extended Integration Layer
#───────────────────────────────────────────────────────────────────────────────

set(EXTENDED_SOURCES
    rbpf_ksc_param_integration.c    # Core: lifecycle, step
    rbpf_ext_config.c               # Configuration functions
    rbpf_ext_diagnostics.c          # Getters and print functions
)

set(EXTENDED_HEADERS
    rbpf_ksc_param_integration.h    # Main header (includes types + api)
    rbpf_ext_types.h                # Structs and enums
    rbpf_ext_api.h                  # Function declarations
)

# Add sources to parent scope for rocks library
set(EXTENDED_SOURCES_FULL "")
foreach(src ${EXTENDED_SOURCES})
    list(APPEND EXTENDED_SOURCES_FULL ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()
set(EXTENDED_SOURCES ${EXTENDED_SOURCES_FULL} PARENT_SCOPE)

# Export include directory to parent
set(EXTENDED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Install headers
install(FILES ${EXTENDED_HEADERS}
    DESTINATION include/rocks/extended
)
