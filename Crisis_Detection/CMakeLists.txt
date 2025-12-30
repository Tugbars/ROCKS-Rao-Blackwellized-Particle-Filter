# CMakeLists.txt for Crisis_Detection subsystem
cmake_minimum_required(VERSION 3.16)
project(crisis_detection VERSION 1.0.0 LANGUAGES C)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
option(USE_MKL "Use Intel MKL for vectorized math" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCH "Build benchmark executable" ON)
option(BUILD_PYTHON_LIB "Build shared library for Python bindings" ON)
set(SIMD_ARCH "native" CACHE STRING "SIMD architecture: native, avx512, avx2, sse4, none")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER FLAGS (only for standalone build, skip if part of parent project)
# ═══════════════════════════════════════════════════════════════════════════════
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Standalone build
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)

    if(NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

        # SIMD flags
        if(SIMD_ARCH STREQUAL "native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        elseif(SIMD_ARCH STREQUAL "avx512")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=skylake-avx512")
        elseif(SIMD_ARCH STREQUAL "avx2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=haswell")
        elseif(SIMD_ARCH STREQUAL "sse4")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.2")
        endif()

        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_FMA)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# FIND MKL (only for standalone build)
# ═══════════════════════════════════════════════════════════════════════════════
if(NOT TARGET MKL::MKL)
    set(MKL_FOUND FALSE)

    if(USE_MKL)
        if(DEFINED ENV{MKLROOT})
            set(MKL_ROOT $ENV{MKLROOT})
        elseif(DEFINED ENV{ONEAPI_ROOT})
            set(MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest")
        elseif(EXISTS "/opt/intel/oneapi/mkl/latest")
            set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
        elseif(EXISTS "/opt/intel/mkl")
            set(MKL_ROOT "/opt/intel/mkl")
        endif()

        if(MKL_ROOT AND EXISTS "${MKL_ROOT}/include/mkl.h")
            set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
            
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
                if(NOT EXISTS "${MKL_LIB_DIR}")
                    set(MKL_LIB_DIR "${MKL_ROOT}/lib")
                endif()
            else()
                set(MKL_LIB_DIR "${MKL_ROOT}/lib/ia32")
            endif()

            find_library(MKL_CORE_LIB mkl_core PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
            find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
            find_library(MKL_SEQUENTIAL_LIB mkl_sequential PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)

            if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB)
                set(MKL_FOUND TRUE)
                set(MKL_LIBRARIES 
                    ${MKL_INTEL_LP64_LIB}
                    ${MKL_SEQUENTIAL_LIB}
                    ${MKL_CORE_LIB}
                )
                message(STATUS "Crisis_Detection: Found MKL at ${MKL_ROOT}")
            endif()
        endif()

        if(NOT MKL_FOUND)
            message(STATUS "Crisis_Detection: MKL not found, using scalar fallback")
        endif()
    endif()
else()
    # Parent project already found MKL
    set(MKL_FOUND TRUE)
    message(STATUS "Crisis_Detection: Using parent MKL::MKL target")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY: hawkes_integrator (standalone - needed by RBPF)
# ═══════════════════════════════════════════════════════════════════════════════
add_library(hawkes_integrator STATIC
    hawkes_signal/hawkes_integrator.c
)

target_include_directories(hawkes_integrator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/hawkes_signal
)

# Platform-specific math library
if(UNIX)
    target_link_libraries(hawkes_integrator PUBLIC m)
endif()

# MKL acceleration
if(TARGET MKL::MKL)
    target_link_libraries(hawkes_integrator PUBLIC MKL::MKL)
    target_compile_definitions(hawkes_integrator PUBLIC USE_MKL=1)
    message(STATUS "  hawkes_integrator - with MKL::MKL")
elseif(MKL_FOUND)
    target_compile_definitions(hawkes_integrator PUBLIC USE_MKL=1)
    target_include_directories(hawkes_integrator PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(hawkes_integrator PUBLIC ${MKL_LIBRARIES})
    if(UNIX)
        target_link_libraries(hawkes_integrator PUBLIC pthread dl)
    endif()
    message(STATUS "  hawkes_integrator - with MKL (manual)")
else()
    message(STATUS "  hawkes_integrator - scalar fallback (no MKL)")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY: sr_detector (standalone)
# ═══════════════════════════════════════════════════════════════════════════════
add_library(sr_detector STATIC
    sr_detector/sr_detector.c
)

target_include_directories(sr_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/sr_detector
)

if(UNIX)
    target_link_libraries(sr_detector PUBLIC m)
endif()

# MKL for sr_detector (uses VML functions)
if(TARGET MKL::MKL)
    target_link_libraries(sr_detector PUBLIC MKL::MKL)
    target_compile_definitions(sr_detector PUBLIC __INTEL_MKL__ USE_MKL=1)
elseif(MKL_FOUND)
    target_compile_definitions(sr_detector PUBLIC __INTEL_MKL__ USE_MKL=1)
    target_include_directories(sr_detector PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(sr_detector PUBLIC ${MKL_LIBRARIES})
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY: event_detector (standalone)
# ═══════════════════════════════════════════════════════════════════════════════
add_library(event_detector STATIC
    event_detector/event_detector.c
)

target_include_directories(event_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/event_detector
)

if(UNIX)
    target_link_libraries(event_detector PUBLIC m)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY: crisis_detection (combined - links to components)
# ═══════════════════════════════════════════════════════════════════════════════
add_library(crisis_detection STATIC
    crisis_detector/crisis_detector.c
)

target_include_directories(crisis_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sr_detector
    ${CMAKE_CURRENT_SOURCE_DIR}/event_detector
    ${CMAKE_CURRENT_SOURCE_DIR}/hawkes_signal
    ${CMAKE_CURRENT_SOURCE_DIR}/crisis_detector
)

# Link to component libraries
target_link_libraries(crisis_detection PUBLIC
    sr_detector
    event_detector
    hawkes_integrator
)

if(TARGET MKL::MKL)
    target_compile_definitions(crisis_detection PUBLIC __INTEL_MKL__ USE_MKL=1)
    target_link_libraries(crisis_detection PUBLIC MKL::MKL)
    message(STATUS "  crisis_detection - with MKL::MKL")
elseif(MKL_FOUND)
    target_compile_definitions(crisis_detection PUBLIC __INTEL_MKL__ USE_MKL=1)
    target_include_directories(crisis_detection PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(crisis_detection PUBLIC ${MKL_LIBRARIES})
    if(UNIX)
        target_link_libraries(crisis_detection PUBLIC pthread dl)
    endif()
    message(STATUS "  crisis_detection - with MKL (manual)")
else()
    message(STATUS "  crisis_detection - scalar fallback (no MKL)")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED LIBRARY: libcrisis (for Python bindings)
# ═══════════════════════════════════════════════════════════════════════════════
if(BUILD_PYTHON_LIB)
    # Source files for shared library
    set(CRISIS_SHARED_SOURCES
        sr_detector/sr_detector.c
        event_detector/event_detector.c
        hawkes_signal/hawkes_integrator.c
        crisis_detector/crisis_detector.c
    )

    # On Windows, add the .def file for exports
    if(WIN32 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libcrisis.def)
        set(CRISIS_SHARED_SOURCES ${CRISIS_SHARED_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/libcrisis.def
        )
        message(STATUS "  libcrisis - using libcrisis.def for exports")
    endif()

    add_library(crisis SHARED ${CRISIS_SHARED_SOURCES})

    target_include_directories(crisis PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/sr_detector
        ${CMAKE_CURRENT_SOURCE_DIR}/event_detector
        ${CMAKE_CURRENT_SOURCE_DIR}/hawkes_signal
        ${CMAKE_CURRENT_SOURCE_DIR}/crisis_detector
    )

    # Platform-specific settings
    if(WIN32)
        # Windows: output as libcrisis.dll
        set_target_properties(crisis PROPERTIES
            OUTPUT_NAME "libcrisis"
            PREFIX ""
            SUFFIX ".dll"
        )
    else()
        # Linux/Mac: output as libcrisis.so
        set_target_properties(crisis PROPERTIES
            OUTPUT_NAME "crisis"
            PREFIX "lib"
        )
        target_link_libraries(crisis PUBLIC m)
    endif()

    # MKL acceleration for shared library
    if(TARGET MKL::MKL)
        target_link_libraries(crisis PUBLIC MKL::MKL)
        target_compile_definitions(crisis PUBLIC __INTEL_MKL__ USE_MKL=1)
        message(STATUS "  libcrisis (Python) - with MKL::MKL")
    elseif(MKL_FOUND)
        target_compile_definitions(crisis PUBLIC __INTEL_MKL__ USE_MKL=1)
        target_include_directories(crisis PUBLIC ${MKL_INCLUDE_DIR})
        target_link_libraries(crisis PUBLIC ${MKL_LIBRARIES})
        if(UNIX)
            target_link_libraries(crisis PUBLIC pthread dl)
        endif()
        message(STATUS "  libcrisis (Python) - with MKL (manual)")
    else()
        message(STATUS "  libcrisis (Python) - scalar fallback (no MKL)")
    endif()

    # Install shared library
    install(TARGETS crisis
        RUNTIME DESTINATION bin      # Windows DLL
        LIBRARY DESTINATION lib      # Linux/Mac .so
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════
if(BUILD_TESTS)
    enable_testing()
    
    # SR Detector tests
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sr_detector/test_sr_detector.c)
        add_executable(test_sr_detector sr_detector/test_sr_detector.c)
        target_link_libraries(test_sr_detector PRIVATE crisis_detection)
        add_test(NAME sr_detector_tests COMMAND test_sr_detector)
    endif()
    
    # Event Detector tests
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/event_detector/test_event_detector.c)
        add_executable(test_event_detector event_detector/test_event_detector.c)
        target_link_libraries(test_event_detector PRIVATE crisis_detection)
        add_test(NAME event_detector_tests COMMAND test_event_detector)
    endif()
    
    # Hawkes scenarios tests
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hawkes_signal/test_hawkes_scenarios.c)
        add_executable(test_hawkes_scenarios hawkes_signal/test_hawkes_scenarios.c)
        target_link_libraries(test_hawkes_scenarios PRIVATE hawkes_integrator)
        add_test(NAME hawkes_scenarios COMMAND test_hawkes_scenarios)
    endif()
    
    # Crisis Detector tests
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/crisis_detector/test_crisis_detector.c)
        add_executable(test_crisis_detector crisis_detector/test_crisis_detector.c)
        target_link_libraries(test_crisis_detector PRIVATE crisis_detection)
        add_test(NAME crisis_detector_tests COMMAND test_crisis_detector)
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
if(BUILD_BENCH)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/crisis_detector/bench_crisis_detector.c)
        add_executable(bench_crisis_detector crisis_detector/bench_crisis_detector.c)
        target_link_libraries(bench_crisis_detector PRIVATE crisis_detection)
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════
install(TARGETS 
    hawkes_integrator 
    sr_detector 
    event_detector 
    crisis_detection
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    hawkes_signal/hawkes_integrator.h
    sr_detector/sr_detector.h
    event_detector/event_detector.h
    event_detector/p2_quantile.h
    crisis_detector/crisis_detector.h
    DESTINATION include/crisis_detection
)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  Crisis Detection Configuration")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD arch:      ${SIMD_ARCH}")
message(STATUS "  MKL:            ${MKL_FOUND}")
message(STATUS "  Build tests:    ${BUILD_TESTS}")
message(STATUS "  Build bench:    ${BUILD_BENCH}")
message(STATUS "  Python lib:     ${BUILD_PYTHON_LIB}")
message(STATUS "")
message(STATUS "  Libraries:")
message(STATUS "    - hawkes_integrator  (standalone, for RBPF)")
message(STATUS "    - sr_detector        (standalone)")
message(STATUS "    - event_detector     (standalone)")
message(STATUS "    - crisis_detection   (combined static)")
if(BUILD_PYTHON_LIB)
message(STATUS "    - libcrisis          (shared, for Python)")
endif()
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")