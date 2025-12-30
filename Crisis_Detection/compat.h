/*
 * compat.h - Cross-platform compatibility for Windows/MSVC and Linux/GCC
 * 
 * Include this at the top of files that use:
 *   - rdtsc() for cycle counting
 *   - aligned_alloc() for SIMD-aligned memory
 *   - alloca() for stack allocation
 */

#ifndef COMPAT_H
#define COMPAT_H

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * RDTSC - Cycle Counter
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(_MSC_VER)
    /* MSVC on Windows */
    #include <intrin.h>
    static inline uint64_t rdtsc(void) {
        return __rdtsc();
    }
#elif defined(__x86_64__) || defined(__i386__)
    /* GCC/Clang on x86 */
    static inline uint64_t rdtsc(void) {
        unsigned int lo, hi;
        __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | lo;
    }
#else
    /* Fallback: use clock_gettime */
    #include <time.h>
    static inline uint64_t rdtsc(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * ALIGNED MEMORY ALLOCATION
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(_MSC_VER)
    /* MSVC on Windows */
    #include <malloc.h>
    #define compat_aligned_alloc(align, size) _aligned_malloc((size), (align))
    #define compat_aligned_free(ptr) _aligned_free(ptr)
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    /* C11 aligned_alloc */
    #include <stdlib.h>
    #define compat_aligned_alloc(align, size) aligned_alloc((align), (size))
    #define compat_aligned_free(ptr) free(ptr)
#else
    /* POSIX posix_memalign fallback */
    #include <stdlib.h>
    static inline void *compat_aligned_alloc(size_t align, size_t size) {
        void *ptr = NULL;
        if (posix_memalign(&ptr, align, size) != 0) return NULL;
        return ptr;
    }
    #define compat_aligned_free(ptr) free(ptr)
#endif

/* Convenience macros (drop-in replacements) */
#ifndef aligned_alloc
    #define aligned_alloc(align, size) compat_aligned_alloc((align), (size))
#endif
#ifndef aligned_free
    #define aligned_free(ptr) compat_aligned_free(ptr)
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * STACK ALLOCATION (alloca)
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(_MSC_VER)
    #include <malloc.h>
    #define compat_alloca(size) _alloca(size)
#else
    #include <alloca.h>
    #define compat_alloca(size) alloca(size)
#endif

/* Drop-in replacement */
#ifndef COMPAT_NO_ALLOCA_MACRO
    #undef alloca
    #define alloca(size) compat_alloca(size)
#endif

#endif /* COMPAT_H */