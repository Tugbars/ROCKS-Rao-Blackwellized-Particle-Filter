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
 *   5. Windows process priority and timer resolution
 *   6. CBWR for reproducible results
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * RECOMMENDED BIOS SETTINGS FOR i9-14900KF HFT:
 * ═══════════════════════════════════════════════════════════════════════════════
 *   - Hyper-Threading: DISABLED (eliminates port contention and jitter)
 *   - E-Cores: DISABLED (ensures L3 cache and ring bus priority)
 *   - CPU Frequency: STATIC (e.g., all-core 5.5-5.7GHz, avoid P-state transitions)
 *   - C-States: DISABLED or C1 only (prevents deep sleep latency)
 *   - Speed Shift: DISABLED (manual frequency control)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WINDOWS POWER SETTINGS:
 * ═══════════════════════════════════════════════════════════════════════════════
 *   PowerShell (run as admin):
 *     # Force scheduler to only use P-cores
 *     powercfg /setacvalueindex scheme_current sub_processor HETEROPOLICY 0
 *     # High performance scheduling
 *     powercfg /setacvalueindex scheme_current sub_processor SCHEDPOLICY 1
 *     # Apply changes
 *     powercfg /setactive scheme_current
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
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#pragma comment(lib, "winmm.lib") /* For timeBeginPeriod */
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
 * For Intel 12th/13th/14th gen (with HT disabled in BIOS):
 *   i9-14900KF: 8 P-cores → pass p_cores=8
 *   i9-13900K:  8 P-cores → pass p_cores=8
 *   i7-13700K:  8 P-cores → pass p_cores=8
 *   i5-13600K:  6 P-cores → pass p_cores=6
 *
 * Why P-cores only?
 *   - E-cores have different latencies, causing scheduling jitter
 *   - Mixed scheduling hurts cache locality
 *   - For latency-critical code, consistency > throughput
 */
static inline void mkl_tuning_set_threads(int p_cores)
{
    int num_threads;

    if (p_cores > 0)
    {
        num_threads = p_cores;
    }
    else
    {
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
    /* Disable MKL's internal dynamic thread scaling - fixed thread count is
     * more predictable for latency-critical code */
    mkl_set_dynamic(0);

    /* CBWR: Conditional Bitwise Reproducibility
     * Ensures identical results across runs (useful for debugging/validation)
     * Slight overhead (~1-2%), comment out for absolute peak performance */
    mkl_cbwr_set(MKL_CBWR_AVX2);

    /* Note: VML mode (VML_EP) is set per-thread in the PARIS hot path
     * because vmlSetMode is thread-local in modern MKL versions */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * WINDOWS-SPECIFIC TUNING
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _MSC_VER

/**
 * Set Windows process priority to HIGH
 * Reduces scheduling interference from background processes
 */
static inline void mkl_tuning_windows_priority(void)
{
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
}

/**
 * Pin current thread to a specific core
 * @param core_id Core index (0-based)
 */
static inline void mkl_tuning_pin_thread(int core_id)
{
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << core_id);
}

/**
 * Set Windows timer resolution to minimum (typically 0.5ms)
 * Critical for accurate sleep/wait operations in HFT
 *
 * IMPORTANT: Call mkl_tuning_timer_end() before program exit!
 */
static inline void mkl_tuning_timer_begin(void)
{
    /* Request 1ms timer resolution (actual minimum varies by system) */
    timeBeginPeriod(1);
}

/**
 * Restore default Windows timer resolution
 * Call this before program exit to be a good citizen
 */
static inline void mkl_tuning_timer_end(void)
{
    timeEndPeriod(1);
}

/**
 * Full Windows HFT initialization
 * - HIGH priority class
 * - Pin main thread to core 0
 * - Set minimum timer resolution
 */
static inline void mkl_tuning_windows_hft_init(void)
{
    mkl_tuning_windows_priority();
    mkl_tuning_pin_thread(0); /* Pin main thread to P-core 0 */
    mkl_tuning_timer_begin();
}

#endif /* _MSC_VER */

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
 *       mkl_tuning_cleanup();   // Cleanup (Windows timer)
 *       return 0;
 *   }
 *
 * Environment variables (set before running):
 *   MKL_NUM_THREADS=8
 *   OMP_NUM_THREADS=8
 *   KMP_AFFINITY=granularity=fine,compact,1,0
 *   KMP_BLOCKTIME=0           (immediate thread yield, saves power/thermal)
 *   KMP_HW_SUBSET=8c          (use 8 cores)
 */
static inline void mkl_tuning_init(int p_cores, int verbose)
{
    /* 1. Flush denormals (do this first!) */
    mkl_tuning_flush_denormals();

    /* 2. Set thread count */
    mkl_tuning_set_threads(p_cores);

    /* 3. Configure MKL (CBWR, dynamic) */
    mkl_tuning_configure();

/* 4. Windows-specific HFT tuning */
#ifdef _MSC_VER
    mkl_tuning_windows_hft_init();
#endif

    if (verbose)
    {
        printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
        printf("║                      MKL TUNING CONFIGURATION                         ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
        printf("  Denormals:     FLUSH TO ZERO (FTZ+DAZ enabled)\n");
        printf("  MKL threads:   %d\n", mkl_get_max_threads());
#ifdef _OPENMP
        printf("  OMP threads:   %d\n", omp_get_max_threads());
#endif
        printf("  MKL dynamic:   %s\n", mkl_get_dynamic() ? "ON" : "OFF");
        printf("  MKL CBWR:      AVX2 (reproducible results)\n");
#ifdef _MSC_VER
        printf("  Windows:       HIGH priority, timer=1ms\n");
#endif

        MKLVersion version;
        mkl_get_version(&version);
        printf("  MKL version:   %d.%d.%d\n", version.MajorVersion,
               version.MinorVersion, version.UpdateVersion);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * AFFINITY HELPERS (Advanced)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Print affinity configuration hints for i9-14900KF
 */
static inline void mkl_tuning_print_affinity_hints(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("THREAD AFFINITY FOR i9-14900KF (8 P-cores, HT & E-cores disabled)\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("Windows PowerShell (run BEFORE your program):\n");
    printf("  $env:KMP_AFFINITY=\"granularity=fine,compact,1,0\"\n");
    printf("  $env:KMP_BLOCKTIME=\"0\"          # Immediate yield (saves thermal)\n");
    printf("  $env:KMP_HW_SUBSET=\"8c\"         # Use 8 cores\n");
    printf("  $env:MKL_NUM_THREADS=\"8\"\n");
    printf("  $env:OMP_NUM_THREADS=\"8\"\n");
    printf("\n");
    printf("Windows CMD:\n");
    printf("  set KMP_AFFINITY=granularity=fine,compact,1,0\n");
    printf("  set KMP_BLOCKTIME=0\n");
    printf("  set KMP_HW_SUBSET=8c\n");
    printf("\n");
    printf("Linux:\n");
    printf("  export KMP_AFFINITY=granularity=fine,compact,1,0\n");
    printf("  export KMP_BLOCKTIME=0\n");
    printf("  taskset -c 0-7 ./your_program   # Pin to cores 0-7\n");
    printf("\n");
    printf("For explicit core mapping (if HT is ON, skip logical siblings):\n");
    printf("  set KMP_AFFINITY=proclist=[0,2,4,6,8,10,12,14],explicit\n");
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BIOS TUNING HINTS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Print BIOS configuration hints for HFT
 */
static inline void mkl_tuning_print_bios_hints(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("BIOS SETTINGS FOR HFT (i9-14900KF)\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("┌─────────────────────────┬─────────────────┬──────────────────────────────┐\n");
    printf("│ Setting                 │ Value           │ Reason                       │\n");
    printf("├─────────────────────────┼─────────────────┼──────────────────────────────┤\n");
    printf("│ Hyper-Threading         │ DISABLED        │ Eliminates port contention   │\n");
    printf("│ E-Cores                 │ DISABLED        │ L3/ring bus priority         │\n");
    printf("│ CPU Frequency           │ STATIC 5.5GHz   │ No P-state transitions       │\n");
    printf("│ C-States                │ C1 only or OFF  │ No deep sleep latency        │\n");
    printf("│ Speed Shift (HWP)       │ DISABLED        │ Manual frequency control     │\n");
    printf("│ Turbo Boost             │ DISABLED        │ Consistent clocks            │\n");
    printf("│ Power Limits (PL1/PL2)  │ Unlimited       │ No thermal throttling        │\n");
    printf("└─────────────────────────┴─────────────────┴──────────────────────────────┘\n");
    printf("\n");
    printf("Windows Power Settings (PowerShell as Admin):\n");
    printf("  # Force P-cores only for scheduling\n");
    printf("  powercfg /setacvalueindex scheme_current sub_processor HETEROPOLICY 0\n");
    printf("  powercfg /setacvalueindex scheme_current sub_processor SCHEDPOLICY 1\n");
    printf("  powercfg /setactive scheme_current\n");
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CLEANUP
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Cleanup function - call before program exit
 */
static inline void mkl_tuning_cleanup(void)
{
#ifdef _MSC_VER
    mkl_tuning_timer_end();
#endif
}

#endif /* MKL_TUNING_H */