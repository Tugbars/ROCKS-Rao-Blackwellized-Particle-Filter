# ═══════════════════════════════════════════════════════════════════════════════
# Hawkes MLE Oracle - Slow Path Calibration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Maximum Likelihood Estimation for Hawkes process parameters.
# Intended to run periodically (every ~1000 ticks) on slow path.
#
# Provides: hawkes_mle_fit(), hawkes_mle_estimate_gamma()
#

set(HAWKES_MLE_SOURCES
    smc2_hawkes_mle.c
)

set(HAWKES_MLE_HEADERS
    smc2_hawkes_mle.h
)

# ─────────────────────────────────────────────────────────────────────────────
# Static Library
# ─────────────────────────────────────────────────────────────────────────────
add_library(hawkes_mle STATIC ${HAWKES_MLE_SOURCES} ${HAWKES_MLE_HEADERS})

target_include_directories(hawkes_mle
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/smc2/oracle>
)

# Math library on Unix
if(UNIX)
    target_link_libraries(hawkes_mle PRIVATE m)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Compiler Settings
# ─────────────────────────────────────────────────────────────────────────────
target_compile_definitions(hawkes_mle
    PRIVATE
        $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>
)

set_target_properties(hawkes_mle PROPERTIES
    OUTPUT_NAME "hawkes_mle"
    DEBUG_POSTFIX "d"
    POSITION_INDEPENDENT_CODE ON
)

# ─────────────────────────────────────────────────────────────────────────────
# IDE Support
# ─────────────────────────────────────────────────────────────────────────────
source_group("Source Files" FILES ${HAWKES_MLE_SOURCES})
source_group("Header Files" FILES ${HAWKES_MLE_HEADERS})
