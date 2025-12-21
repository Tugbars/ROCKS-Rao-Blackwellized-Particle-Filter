/**
 * @file pgas_omp_compat.h
 * @brief OpenMP compatibility macros for MSVC
 *
 * MSVC requires /openmp:experimental for #pragma omp simd.
 * This header provides conditional macros that work across platforms.
 */

#ifndef PGAS_OMP_COMPAT_H
#define PGAS_OMP_COMPAT_H

#include <omp.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * SIMD PRAGMA COMPATIBILITY
 *
 * MSVC: Requires /openmp:experimental or /openmp:llvm for simd
 * GCC/Clang: Standard OpenMP simd support
 *═══════════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER) && !defined(_OPENMP_SIMD)
    /* MSVC without experimental: disable simd pragma */
    #define PGAS_OMP_SIMD
    #define PGAS_OMP_SIMD_REDUCTION(op, var)
#else
    /* GCC/Clang or MSVC with experimental */
    #define PGAS_OMP_SIMD _Pragma("omp simd")
    #define PGAS_OMP_SIMD_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * ALIGNMENT COMPATIBILITY
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _MSC_VER
    #define PGAS_ALIGNED(x) __declspec(align(x))
    #define PGAS_ALIGN_ATTR
#else
    #define PGAS_ALIGNED(x)
    #define PGAS_ALIGN_ATTR __attribute__((aligned(64)))
#endif

#endif /* PGAS_OMP_COMPAT_H */
