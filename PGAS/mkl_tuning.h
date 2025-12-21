/**
 * @file mkl_tuning.h
 * @brief MKL and CPU tuning for HFT-grade performance
 *
 * Call mkl_tuning_init() at the start of main() before any MKL calls.
 *
 * Key optimizations:
 *   1. Flush denormals to zero (100x speedup for edge cases)
 *   2. P-cores only on hybrid CPUs (avoid E-core scheduling jitter)
 *   3. Thread affinity for cache locality
 *   4. MKL threading layer configuration
 */

#ifndef MKL_TUNING_H
#define MKL_TUNING_H

#include <mkl.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* SSE/AVX control for denormals */
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <xmmintrin.h>
    #include <pmmintrin.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * DENORMAL FLUSHING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Flush denormals to zero - critical for HFT!
 * Denormal operations can be 100x slower than normal floats.
 */
static inline void mkl_tuning_flush_denormals(void)
{
    /* FTZ: Flush To Zero - denormal results become zero */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    
    /* DAZ: Denormals Are Zero - denormal inputs treated as zero */
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * THREAD CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Configure threading for hybrid CPUs (P-cores + E-cores)
 *
 * @param p_cores Number of performance cores (0 = auto-detect/use all)
 *
 * For Intel 12th/13th/14th gen:
 *   i9-13900K: 8 P-cores, 16 E-cores → pass p_cores=8
 *   i7-13700K: 8 P-cores, 8 E-cores  → pass p_cores=8
 *   i5-13600K: 6 P-cores, 8 E-cores  → pass p_cores=6
 *
 * Why P-cores only?
 *   - E-cores have different latencies, causing scheduling jitter
 *   - Mixed scheduling hurts cache locality
 *   - For latency-critical code, consistency > throughput
 */
static inline void mkl_tuning_set_threads(int p_cores)
{
    int num_threads;
    
    if (p_cores > 0) {
        num_threads = p_cores;
    } else {
        /* Default: use all available threads */
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #else
        num_threads = 1;
        #endif
    }
    
    /* Set MKL thread count */
    mkl_set_num_threads(num_threads);
    
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MKL CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Configure MKL for maximum performance
 */
static inline void mkl_tuning_configure(void)
{
    /* Use Intel threading layer (best for Intel CPUs) */
    /* mkl_set_threading_layer(MKL_THREADING_INTEL); */
    
    /* Enable dynamic thread adjustment (can help or hurt) */
    mkl_set_dynamic(0);  /* 0 = fixed thread count, more predictable */
    
    /* Memory alignment hints */
    /* MKL already uses 64-byte alignment internally */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN INIT FUNCTION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize all MKL tuning settings
 *
 * @param p_cores Number of P-cores (0 = use all threads)
 * @param verbose Print configuration info
 *
 * Usage:
 *   int main(void) {
 *       mkl_tuning_init(8, 1);  // 8 P-cores, verbose
 *       // ... rest of program
 *   }
 *
 * Environment variables (alternative to code):
 *   MKL_NUM_THREADS=8
 *   OMP_NUM_THREADS=8
 *   KMP_AFFINITY=granularity=fine,compact,1,0
 *   KMP_HW_SUBSET=8c  (use 8 cores)
 */
static inline void mkl_tuning_init(int p_cores, int verbose)
{
    /* 1. Flush denormals */
    mkl_tuning_flush_denormals();
    
    /* 2. Set thread count */
    mkl_tuning_set_threads(p_cores);
    
    /* 3. Configure MKL */
    mkl_tuning_configure();
    
    if (verbose) {
        printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
        printf("║                      MKL TUNING CONFIGURATION                         ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
        printf("  Denormals:     FLUSH TO ZERO (FTZ+DAZ enabled)\n");
        printf("  MKL threads:   %d\n", mkl_get_max_threads());
        #ifdef _OPENMP
        printf("  OMP threads:   %d\n", omp_get_max_threads());
        #endif
        printf("  MKL dynamic:   %s\n", mkl_get_dynamic() ? "ON" : "OFF");
        
        MKLVersion version;
        mkl_get_version(&version);
        printf("  MKL version:   %d.%d.%d\n", version.MajorVersion, 
               version.MinorVersion, version.UpdateVersion);
        printf("\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * AFFINITY HELPERS (Advanced)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Print affinity configuration hints
 */
static inline void mkl_tuning_print_affinity_hints(void)
{
    printf("Thread Affinity Tips (set before running):\n");
    printf("  Windows CMD:\n");
    printf("    set KMP_AFFINITY=granularity=fine,compact,1,0\n");
    printf("    set KMP_HW_SUBSET=8c   (for 8 P-cores)\n");
    printf("  Windows PowerShell:\n");
    printf("    $env:KMP_AFFINITY=\"granularity=fine,compact,1,0\"\n");
    printf("    $env:KMP_HW_SUBSET=\"8c\"\n");
    printf("  Linux:\n");
    printf("    export KMP_AFFINITY=granularity=fine,compact,1,0\n");
    printf("    taskset -c 0-7 ./your_program  (pin to cores 0-7)\n");
    printf("\n");
}

#endif /* MKL_TUNING_H */
