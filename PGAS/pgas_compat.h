/**
 * @file pgas_compat.h
 * @brief Cross-platform compatibility for atomics
 *
 * MSVC doesn't properly support C11 <stdatomic.h> in C mode.
 * This header provides a compatibility layer using:
 *   - Windows Interlocked functions on MSVC
 *   - C11 atomics on GCC/Clang
 */

#ifndef PGAS_COMPAT_H
#define PGAS_COMPAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * PLATFORM DETECTION
 *═══════════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER)
    #define PGAS_PLATFORM_MSVC 1
#elif defined(__GNUC__) || defined(__clang__)
    #define PGAS_PLATFORM_GCC 1
#else
    #error "Unsupported compiler"
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * MSVC: Use Windows Interlocked Functions
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef PGAS_PLATFORM_MSVC

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>

/* Atomic types */
typedef volatile LONG        pgas_atomic_int32;
typedef volatile LONG64      pgas_atomic_int64;

/* Initialization */
#define PGAS_ATOMIC_INIT(val) (val)

/* Load (acquire semantics) */
static inline int32_t pgas_atomic_load_32(pgas_atomic_int32* ptr) {
    int32_t val = *ptr;
    _ReadBarrier();
    return val;
}

static inline int64_t pgas_atomic_load_64(pgas_atomic_int64* ptr) {
    int64_t val = *ptr;
    _ReadBarrier();
    return val;
}

/* Store (release semantics) */
static inline void pgas_atomic_store_32(pgas_atomic_int32* ptr, int32_t val) {
    _WriteBarrier();
    *ptr = val;
}

static inline void pgas_atomic_store_64(pgas_atomic_int64* ptr, int64_t val) {
    _WriteBarrier();
    *ptr = val;
}

/* Fetch-add (returns old value) */
static inline int32_t pgas_atomic_fetch_add_32(pgas_atomic_int32* ptr, int32_t val) {
    return InterlockedExchangeAdd(ptr, val);
}

static inline int64_t pgas_atomic_fetch_add_64(pgas_atomic_int64* ptr, int64_t val) {
    return InterlockedExchangeAdd64(ptr, val);
}

/* Compare-exchange (returns true if swapped) */
static inline int pgas_atomic_cas_32(pgas_atomic_int32* ptr, int32_t* expected, int32_t desired) {
    int32_t old = InterlockedCompareExchange(ptr, desired, *expected);
    if (old == *expected) {
        return 1;
    } else {
        *expected = old;
        return 0;
    }
}

/* Memory fence */
static inline void pgas_atomic_fence(void) {
    MemoryBarrier();
}

#endif /* PGAS_PLATFORM_MSVC */

/*═══════════════════════════════════════════════════════════════════════════════
 * GCC/Clang: Use C11 Atomics
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef PGAS_PLATFORM_GCC

#include <stdatomic.h>

/* Atomic types */
typedef _Atomic int32_t      pgas_atomic_int32;
typedef _Atomic int64_t      pgas_atomic_int64;

/* Initialization */
#define PGAS_ATOMIC_INIT(val) (val)

/* Load (acquire semantics) */
static inline int32_t pgas_atomic_load_32(pgas_atomic_int32* ptr) {
    return atomic_load_explicit(ptr, memory_order_acquire);
}

static inline int64_t pgas_atomic_load_64(pgas_atomic_int64* ptr) {
    return atomic_load_explicit(ptr, memory_order_acquire);
}

/* Store (release semantics) */
static inline void pgas_atomic_store_32(pgas_atomic_int32* ptr, int32_t val) {
    atomic_store_explicit(ptr, val, memory_order_release);
}

static inline void pgas_atomic_store_64(pgas_atomic_int64* ptr, int64_t val) {
    atomic_store_explicit(ptr, val, memory_order_release);
}

/* Fetch-add (returns old value) */
static inline int32_t pgas_atomic_fetch_add_32(pgas_atomic_int32* ptr, int32_t val) {
    return atomic_fetch_add_explicit(ptr, val, memory_order_acq_rel);
}

static inline int64_t pgas_atomic_fetch_add_64(pgas_atomic_int64* ptr, int64_t val) {
    return atomic_fetch_add_explicit(ptr, val, memory_order_acq_rel);
}

/* Compare-exchange (returns true if swapped) */
static inline int pgas_atomic_cas_32(pgas_atomic_int32* ptr, int32_t* expected, int32_t desired) {
    return atomic_compare_exchange_strong_explicit(ptr, expected, desired,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire);
}

/* Memory fence */
static inline void pgas_atomic_fence(void) {
    atomic_thread_fence(memory_order_seq_cst);
}

#endif /* PGAS_PLATFORM_GCC */

#ifdef __cplusplus
}
#endif

#endif /* PGAS_COMPAT_H */
