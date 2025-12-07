/*
 * MKL Configuration for Low-Latency Quant Trading
 *
 * Optimized for Intel Hybrid CPUs (12th-14th Gen) and AMD Ryzen/EPYC
 *
 * KEY OPTIMIZATIONS FOR <200µs LATENCY:
 *   1. P-cores only, NO hyperthreading (1t per core, not 2t)
 *   2. Infinite blocktime (threads never sleep, burn CPU)
 *   3. DAZ/FTZ enabled (flush denormals to zero)
 *   4. Core affinity pinning (avoid E-cores and scheduler jitter)
 *
 * Usage:
 *   #include "mkl_config.h"
 *
 *   int main() {
 *       mkl_config_quant_mode(1);  // 1 = verbose
 *       // ... your code ...
 *   }
 *
 * OR set environment variables in launch script (more reliable):
 *
 *   export KMP_AFFINITY="granularity=fine,compact,1,0"
 *   export KMP_HW_SUBSET="1s,8c,1t"
 *   export KMP_BLOCKTIME="infinite"
 *   export KMP_LIBRARY="turnaround"
 *   export MKL_ENABLE_INSTRUCTIONS="AVX2"
 *   export MKL_NUM_THREADS=8
 *   ./quant_bot
 */

#ifndef MKL_CONFIG_H
#define MKL_CONFIG_H

#ifdef SSA_USE_MKL

#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* SIMD intrinsics for DAZ/FTZ */
#include <xmmintrin.h>
#include <pmmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Platform-specific macros
 * ============================================================================ */

#ifdef _WIN32
    #include <intrin.h>
    #define MKL_SETENV(name, value) _putenv_s(name, value)
#else
    #define MKL_SETENV(name, value) setenv(name, value, 1)
#endif

/* ============================================================================
 * CPU Detection
 * ============================================================================ */

typedef struct {
    int num_p_cores;         /* Performance cores (physical) */
    int num_e_cores;         /* Efficiency cores (physical) */
    int total_logical;       /* Total logical processors */
    int is_hybrid;           /* 1 if P+E cores detected */
    int has_avx512;
    int has_avx2;
    int l3_cache_kb;
    char cpu_name[64];
    char vendor[16];         /* "GenuineIntel" or "AuthenticAMD" */
} CPUInfo;

static inline void mkl_cpuid(int info[4], int leaf, int subleaf)
{
#if defined(_MSC_VER) || defined(_WIN32)
    __cpuidex(info, leaf, subleaf);
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(leaf), "c"(subleaf));
#endif
}

static CPUInfo detect_cpu(void)
{
    CPUInfo info = {0};
    int regs[4];

    /* Get vendor string */
    mkl_cpuid(regs, 0, 0);
    memcpy(info.vendor, &regs[1], 4);
    memcpy(info.vendor + 4, &regs[3], 4);
    memcpy(info.vendor + 8, &regs[2], 4);
    info.vendor[12] = '\0';

    /* Get CPU brand string */
    mkl_cpuid(regs, 0x80000002, 0);
    memcpy(info.cpu_name, regs, 16);
    mkl_cpuid(regs, 0x80000003, 0);
    memcpy(info.cpu_name + 16, regs, 16);
    mkl_cpuid(regs, 0x80000004, 0);
    memcpy(info.cpu_name + 32, regs, 16);

    /* Check for AVX-512 and AVX2 */
    mkl_cpuid(regs, 7, 0);
    info.has_avx512 = (regs[1] >> 16) & 1;
    info.has_avx2 = (regs[1] >> 5) & 1;

    /* Get L3 cache size */
    mkl_cpuid(regs, 0x80000006, 0);
    info.l3_cache_kb = ((regs[2] >> 18) & 0x3FFF) * 512;

    /* Detect Intel hybrid architecture (12th gen+) */
    int is_intel = (strstr(info.vendor, "Intel") != NULL);
    int is_amd = (strstr(info.vendor, "AMD") != NULL);

    mkl_cpuid(regs, 0, 0);
    int max_leaf = regs[0];

    if (is_intel && max_leaf >= 0x1A) {
        mkl_cpuid(regs, 0x1A, 0);
        int core_type = (regs[0] >> 24) & 0xFF;
        
        /* Detect based on CPU model */
        if (strstr(info.cpu_name, "14900") || strstr(info.cpu_name, "14700")) {
            info.is_hybrid = 1;
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.l3_cache_kb = 36 * 1024;
        }
        else if (strstr(info.cpu_name, "13900") || strstr(info.cpu_name, "13700")) {
            info.is_hybrid = 1;
            info.num_p_cores = 8;
            info.num_e_cores = 16;
            info.l3_cache_kb = 36 * 1024;
        }
        else if (strstr(info.cpu_name, "12900") || strstr(info.cpu_name, "12700")) {
            info.is_hybrid = 1;
            info.num_p_cores = 8;
            info.num_e_cores = 8;
            info.l3_cache_kb = 30 * 1024;
        }
        else if (strstr(info.cpu_name, "12600") || strstr(info.cpu_name, "12400")) {
            info.is_hybrid = 1;
            info.num_p_cores = 6;
            info.num_e_cores = 4;
            info.l3_cache_kb = 20 * 1024;
        }
        else if (core_type == 0x40 || core_type == 0x20) {
            /* Generic hybrid detection */
            info.is_hybrid = 1;
            info.num_p_cores = 6;
            info.num_e_cores = 8;
        }
    }
    else if (is_amd) {
        /* AMD Ryzen / EPYC - no hybrid, all cores equal */
        info.is_hybrid = 0;
        
        if (strstr(info.cpu_name, "9950X") || strstr(info.cpu_name, "9900X")) {
            info.num_p_cores = 16;
            info.num_e_cores = 0;
            info.l3_cache_kb = 64 * 1024;
        }
        else if (strstr(info.cpu_name, "9700X") || strstr(info.cpu_name, "9600X")) {
            info.num_p_cores = 8;
            info.num_e_cores = 0;
            info.l3_cache_kb = 32 * 1024;
        }
        else if (strstr(info.cpu_name, "EPYC") && strstr(info.cpu_name, "9575")) {
            info.num_p_cores = 64;
            info.num_e_cores = 0;
            info.l3_cache_kb = 256 * 1024;
        }
        else if (strstr(info.cpu_name, "EPYC")) {
            info.num_p_cores = 64;
            info.num_e_cores = 0;
            info.l3_cache_kb = 256 * 1024;
        }
        else if (strstr(info.cpu_name, "7950X") || strstr(info.cpu_name, "7900X")) {
            info.num_p_cores = 16;
            info.num_e_cores = 0;
            info.l3_cache_kb = 64 * 1024;
        }
        else {
            /* Generic AMD */
            mkl_cpuid(regs, 1, 0);
            int logical = (regs[1] >> 16) & 0xFF;
            info.num_p_cores = logical / 2;  /* Assume SMT */
            info.num_e_cores = 0;
        }
    }
    else {
        /* Unknown/older CPU */
        info.is_hybrid = 0;
        mkl_cpuid(regs, 1, 0);
        int logical = (regs[1] >> 16) & 0xFF;
        info.num_p_cores = logical / 2;
        info.num_e_cores = 0;
    }

    /* Calculate total logical processors */
    if (info.is_hybrid) {
        /* Intel hybrid: P-cores have HT, E-cores don't */
        info.total_logical = info.num_p_cores * 2 + info.num_e_cores;
    } else {
        /* AMD or non-hybrid Intel: assume SMT-2 */
        info.total_logical = info.num_p_cores * 2;
    }

    return info;
}

/* ============================================================================
 * Configuration Modes
 * ============================================================================ */

typedef enum {
    MKL_CONFIG_AUTO,         /* Auto-detect optimal settings */
    MKL_CONFIG_P_CORES,      /* P-cores only (recommended for latency) */
    MKL_CONFIG_ALL_CORES,    /* All cores (throughput mode) */
    MKL_CONFIG_SEQUENTIAL,   /* Single-threaded */
    MKL_CONFIG_CUSTOM        /* Use environment variables */
} MKLConfigMode;

typedef enum {
    MKL_LATENCY_MODE,        /* Optimize for low latency (quant trading) */
    MKL_THROUGHPUT_MODE      /* Optimize for throughput (backtesting) */
} MKLOptimizationMode;

typedef struct {
    MKLConfigMode mode;
    MKLOptimizationMode opt_mode;
    int num_threads;
    int verbose;
    int daz_ftz_enabled;
    CPUInfo cpu;
} MKLConfig;

static MKLConfig g_mkl_config = {0};

/* ============================================================================
 * Denormal Handling (DAZ/FTZ)
 * 
 * CRITICAL FOR LOW LATENCY:
 * Denormal numbers (very small, ~1e-310) trigger microcode traps
 * that can take 100-1500 cycles instead of ~4 cycles.
 * 
 * DAZ = Denormals Are Zero (treat denormal inputs as zero)
 * FTZ = Flush To Zero (flush denormal results to zero)
 * ============================================================================ */

static inline void mkl_config_enable_daz_ftz(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    g_mkl_config.daz_ftz_enabled = 1;
}

static inline void mkl_config_disable_daz_ftz(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    g_mkl_config.daz_ftz_enabled = 0;
}

/* ============================================================================
 * Thread Affinity Configuration
 * ============================================================================ */

/*
 * Configure for P-cores only, NO hyperthreading.
 * 
 * WHY NO HYPERTHREADING FOR LOW LATENCY:
 *   - HT shares L1 cache, execution units between 2 threads
 *   - Causes cache evictions and resource contention
 *   - Increases latency variance (jitter)
 *   - 1 thread per physical core = predictable, stable latency
 *
 * KMP_HW_SUBSET="1s,8c,1t":
 *   - 1s = 1 socket
 *   - 8c = 8 cores (P-cores only on hybrid)
 *   - 1t = 1 thread per core (NO hyperthreading)
 */
static void mkl_config_set_p_core_affinity_no_ht(int num_p_cores)
{
    char subset[32];
    snprintf(subset, sizeof(subset), "1s,%dc,1t", num_p_cores);
    
    MKL_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");
    MKL_SETENV("KMP_HW_SUBSET", subset);
}

/*
 * Configure for P-cores with hyperthreading (throughput mode).
 * Use for backtesting, not live trading.
 */
static void mkl_config_set_p_core_affinity_with_ht(int num_p_cores)
{
    char subset[32];
    snprintf(subset, sizeof(subset), "1s,%dc,2t", num_p_cores);
    
    MKL_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");
    MKL_SETENV("KMP_HW_SUBSET", subset);
}

/*
 * Configure for all cores (E-cores included).
 * NOT recommended for latency-sensitive code.
 */
static void mkl_config_set_all_core_affinity(void)
{
    MKL_SETENV("KMP_AFFINITY", "granularity=fine,scatter");
}

/*
 * Disable affinity (let OS schedule).
 */
static void mkl_config_disable_affinity(void)
{
    MKL_SETENV("KMP_AFFINITY", "disabled");
}

/* ============================================================================
 * Thread Sleep/Wake Configuration
 * 
 * CRITICAL FOR LOW LATENCY:
 * Default MKL behavior: threads sleep after ~200ms idle
 * Waking a sleeping thread costs 10-30µs (OS scheduler overhead)
 * 
 * For <200µs latency budget, this is unacceptable.
 * ============================================================================ */

/*
 * Latency mode: threads NEVER sleep, spin-wait forever.
 * Burns 100% CPU on worker threads even when idle.
 * Eliminates wake-up latency entirely.
 */
static void mkl_config_set_latency_mode(void)
{
    MKL_SETENV("KMP_BLOCKTIME", "infinite");
    MKL_SETENV("KMP_LIBRARY", "turnaround");
    g_mkl_config.opt_mode = MKL_LATENCY_MODE;
}

/*
 * Throughput mode: threads sleep when idle.
 * Better for power efficiency and shared systems.
 * Use for backtesting, research, not live trading.
 */
static void mkl_config_set_throughput_mode(void)
{
    MKL_SETENV("KMP_BLOCKTIME", "200");
    MKL_SETENV("KMP_LIBRARY", "throughput");
    g_mkl_config.opt_mode = MKL_THROUGHPUT_MODE;
}

/* ============================================================================
 * Instruction Set Configuration
 * ============================================================================ */

static void mkl_config_set_instructions(const char* level)
{
    if (strcmp(level, "AVX512") == 0) {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX512");
    }
    else if (strcmp(level, "AVX2") == 0) {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");
    }
    else if (strcmp(level, "AVX") == 0) {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX");
    }
    else if (strcmp(level, "SSE4_2") == 0) {
        MKL_SETENV("MKL_ENABLE_INSTRUCTIONS", "SSE4_2");
    }
}

/* ============================================================================
 * Main Initialization Functions
 * ============================================================================ */

/*
 * Initialize with explicit settings.
 */
static int mkl_config_init_ex(MKLConfigMode mode, MKLOptimizationMode opt_mode, int verbose)
{
    g_mkl_config.cpu = detect_cpu();
    g_mkl_config.mode = mode;
    g_mkl_config.opt_mode = opt_mode;
    g_mkl_config.verbose = verbose;

    CPUInfo* cpu = &g_mkl_config.cpu;

    if (verbose) {
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║              MKL CONFIGURATION                            ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  CPU: %-52s ║\n", cpu->cpu_name);
        printf("║  Vendor: %-49s ║\n", cpu->vendor);
        printf("║  Hybrid: %-49s ║\n", cpu->is_hybrid ? "Yes (P+E cores)" : "No");
        if (cpu->is_hybrid) {
            printf("║  P-cores: %-48d ║\n", cpu->num_p_cores);
            printf("║  E-cores: %-48d ║\n", cpu->num_e_cores);
        } else {
            printf("║  Cores: %-50d ║\n", cpu->num_p_cores);
        }
        printf("║  L3 Cache: %-47d KB ║\n", cpu->l3_cache_kb);
        printf("║  AVX-512: %-48s ║\n", cpu->has_avx512 ? "Yes" : "No");
        printf("║  AVX2: %-51s ║\n", cpu->has_avx2 ? "Yes" : "No");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
    }

    /* Determine thread count based on mode */
    int num_threads;
    
    switch (mode) {
    case MKL_CONFIG_SEQUENTIAL:
        num_threads = 1;
        break;
        
    case MKL_CONFIG_P_CORES:
        /* 
         * For LATENCY mode: 1 thread per P-core (NO hyperthreading)
         * For THROUGHPUT mode: 2 threads per P-core (with hyperthreading)
         */
        if (opt_mode == MKL_LATENCY_MODE) {
            num_threads = cpu->num_p_cores;
            mkl_config_set_p_core_affinity_no_ht(cpu->num_p_cores);
        } else {
            num_threads = cpu->num_p_cores * 2;
            mkl_config_set_p_core_affinity_with_ht(cpu->num_p_cores);
        }
        break;
        
    case MKL_CONFIG_ALL_CORES:
        num_threads = cpu->total_logical;
        mkl_config_set_all_core_affinity();
        break;
        
    case MKL_CONFIG_CUSTOM:
        {
            char* env = getenv("MKL_NUM_THREADS");
            if (env) {
                num_threads = atoi(env);
            } else {
                num_threads = cpu->num_p_cores;
            }
        }
        break;
        
    case MKL_CONFIG_AUTO:
    default:
        if (cpu->is_hybrid) {
            if (opt_mode == MKL_LATENCY_MODE) {
                num_threads = cpu->num_p_cores;
                mkl_config_set_p_core_affinity_no_ht(cpu->num_p_cores);
            } else {
                num_threads = cpu->num_p_cores * 2;
                mkl_config_set_p_core_affinity_with_ht(cpu->num_p_cores);
            }
        } else {
            /* Non-hybrid (AMD, older Intel) */
            if (opt_mode == MKL_LATENCY_MODE) {
                num_threads = cpu->num_p_cores;  /* 1 per physical core */
            } else {
                num_threads = cpu->num_p_cores * 2;  /* With SMT */
            }
        }
        break;
    }

    g_mkl_config.num_threads = num_threads;

    /* Set thread blocking behavior */
    if (opt_mode == MKL_LATENCY_MODE) {
        mkl_config_set_latency_mode();
    } else {
        mkl_config_set_throughput_mode();
    }

    /* Configure MKL */
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    if (verbose) {
        printf("║  Mode: %-51s ║\n", 
               opt_mode == MKL_LATENCY_MODE ? "LATENCY (no HT, no sleep)" : "THROUGHPUT (HT, sleep OK)");
        printf("║  MKL Threads: %-44d ║\n", num_threads);
        printf("║  Hyperthreading: %-41s ║\n", 
               (opt_mode == MKL_LATENCY_MODE) ? "DISABLED" : "ENABLED");
        printf("║  Thread Sleep: %-43s ║\n",
               (opt_mode == MKL_LATENCY_MODE) ? "NEVER (busy spin)" : "ENABLED");
        printf("║  Dynamic Threading: %-38s ║\n", "DISABLED");
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }

    return 0;
}

/*
 * Default initialization (auto-detect, throughput mode).
 */
static int mkl_config_init(void)
{
    return mkl_config_init_ex(MKL_CONFIG_AUTO, MKL_THROUGHPUT_MODE, 0);
}

/*
 * Verbose initialization.
 */
static int mkl_config_init_verbose(void)
{
    return mkl_config_init_ex(MKL_CONFIG_AUTO, MKL_THROUGHPUT_MODE, 1);
}

/* ============================================================================
 * QUANT MODE - The main function for trading applications
 * 
 * Combines all low-latency optimizations:
 *   1. P-cores only (avoid slow E-cores)
 *   2. NO hyperthreading (1 thread per physical core)
 *   3. Infinite blocktime (threads never sleep)
 *   4. DAZ/FTZ enabled (no denormal traps)
 *   5. AVX2 instructions (AVX-512 disabled on consumer Intel)
 * 
 * Call this ONCE at program start, BEFORE any MKL operations.
 * ============================================================================ */

static void mkl_config_quant_mode(int verbose)
{
    CPUInfo cpu = detect_cpu();
    g_mkl_config.cpu = cpu;
    g_mkl_config.verbose = verbose;
    g_mkl_config.mode = MKL_CONFIG_P_CORES;
    g_mkl_config.opt_mode = MKL_LATENCY_MODE;

    /* 1. Thread affinity: P-cores only, NO hyperthreading */
    mkl_config_set_p_core_affinity_no_ht(cpu.num_p_cores);

    /* 2. Thread sleep: NEVER (busy spin) */
    mkl_config_set_latency_mode();

    /* 3. Instruction set */
    if (cpu.is_hybrid && strstr(cpu.vendor, "Intel")) {
        /* Intel 12th-14th gen: AVX-512 is DISABLED in microcode */
        mkl_config_set_instructions("AVX2");
    } else if (strstr(cpu.vendor, "AMD") && cpu.has_avx512) {
        /* AMD EPYC Zen4: AVX-512 works */
        mkl_config_set_instructions("AVX512");
    } else if (cpu.has_avx2) {
        mkl_config_set_instructions("AVX2");
    }

    /* 4. DAZ/FTZ: Flush denormals to zero */
    mkl_config_enable_daz_ftz();

    /* 5. Configure MKL threads */
    int num_threads = cpu.num_p_cores;  /* 1 thread per physical core */
    g_mkl_config.num_threads = num_threads;
    
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    /* 6. Memory settings */
    MKL_SETENV("MKL_FAST_MEMORY_LIMIT", "0");

    if (verbose) {
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║              QUANT MODE ENGAGED                           ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  CPU: %-52s ║\n", cpu.cpu_name);
        printf("║  Cores: %d P-cores (E-cores ignored)%*s║\n", 
               cpu.num_p_cores, 26 - (cpu.num_p_cores >= 10 ? 1 : 0), "");
        printf("║  Threads: %d (1 per physical core, NO HT)%*s║\n", 
               num_threads, 19 - (num_threads >= 10 ? 1 : 0), "");
        printf("║  Hyperthreading: DISABLED%33s║\n", "");
        printf("║  Thread Sleep: NEVER (infinite blocktime)%17s║\n", "");
        printf("║  DAZ/FTZ: ENABLED (denormals → zero)%22s║\n", "");
        printf("║  Instructions: %-43s ║\n", 
               (cpu.is_hybrid && strstr(cpu.vendor, "Intel")) ? "AVX2" :
               (cpu.has_avx512 ? "AVX-512" : "AVX2"));
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  WARNING: Threads will burn 100%% CPU even when idle.     ║\n");
        printf("║           This is intentional for minimum latency.        ║\n");
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }
}

/*
 * Throughput mode for backtesting/research.
 * Uses hyperthreading and allows thread sleep.
 */
static void mkl_config_throughput_mode(int verbose)
{
    CPUInfo cpu = detect_cpu();
    g_mkl_config.cpu = cpu;
    g_mkl_config.verbose = verbose;
    g_mkl_config.mode = MKL_CONFIG_P_CORES;
    g_mkl_config.opt_mode = MKL_THROUGHPUT_MODE;

    /* P-cores with hyperthreading */
    mkl_config_set_p_core_affinity_with_ht(cpu.num_p_cores);
    mkl_config_set_throughput_mode();

    /* Instruction set */
    if (cpu.is_hybrid && strstr(cpu.vendor, "Intel")) {
        mkl_config_set_instructions("AVX2");
    } else if (cpu.has_avx512) {
        mkl_config_set_instructions("AVX512");
    } else {
        mkl_config_set_instructions("AVX2");
    }

    /* DAZ/FTZ still useful for numerical stability */
    mkl_config_enable_daz_ftz();

    /* 2 threads per P-core (hyperthreading) */
    int num_threads = cpu.num_p_cores * 2;
    g_mkl_config.num_threads = num_threads;
    
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    if (verbose) {
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║              THROUGHPUT MODE                              ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  CPU: %-52s ║\n", cpu.cpu_name);
        printf("║  Threads: %d (with hyperthreading)%*s║\n",
               num_threads, 24 - (num_threads >= 10 ? 1 : 0), "");
        printf("║  Thread Sleep: ENABLED (power efficient)%18s║\n", "");
        printf("║  DAZ/FTZ: ENABLED%41s║\n", "");
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }
}

/* ============================================================================
 * Legacy compatibility functions
 * ============================================================================ */

/* Original function name - now calls quant_mode */
static inline void mkl_config_14900kf(int verbose)
{
    mkl_config_quant_mode(verbose);
}

/* Original SSA-specific setup - now calls quant_mode */
static void mkl_config_ssa_full(int verbose)
{
    mkl_config_quant_mode(verbose);
    
    if (verbose) {
        printf("=== SSA Configuration ===\n");
        printf("Recommendation: Use randomized SVD for k < L/2\n");
        printf("Memory alignment: 64 bytes\n");
        printf("=========================\n\n");
    }
}

/* Generic setup for unknown CPUs */
static void mkl_config_generic(int verbose)
{
    mkl_config_disable_affinity();
    mkl_config_enable_daz_ftz();
    mkl_config_init_ex(MKL_CONFIG_AUTO, MKL_THROUGHPUT_MODE, verbose);
}

/* ============================================================================
 * Runtime Thread Control
 * ============================================================================ */

static void mkl_config_set_threads(int n)
{
    mkl_set_num_threads(n);
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

static void mkl_config_restore_threads(void)
{
    mkl_set_num_threads(g_mkl_config.num_threads);
#ifdef _OPENMP
    omp_set_num_threads(g_mkl_config.num_threads);
#endif
}

static int mkl_config_get_threads(void)
{
    return g_mkl_config.num_threads;
}

/* ============================================================================
 * SSA-Specific Thread Tuning
 * ============================================================================ */

typedef enum {
    SSA_WORKLOAD_DECOMPOSE,
    SSA_WORKLOAD_RECONSTRUCT,
    SSA_WORKLOAD_WCORR,
    SSA_WORKLOAD_FORECAST
} SSAWorkloadType;

static int mkl_config_ssa_threads(int N, int L, int k, SSAWorkloadType workload)
{
    int p_cores = g_mkl_config.cpu.num_p_cores;
    if (p_cores == 0) p_cores = 8;

    /* In latency mode, we use 1 thread per core (no HT) */
    int max_threads = (g_mkl_config.opt_mode == MKL_LATENCY_MODE) 
                      ? p_cores 
                      : p_cores * 2;

    int fft_len = 1;
    while (fft_len < N) fft_len <<= 1;

    switch (workload) {
    case SSA_WORKLOAD_DECOMPOSE:
        if (L < 1024) return 1;
        if (L < 4096) return 2;
        if (L < 16384) return 4;
        return (max_threads > 8) ? 8 : max_threads;

    case SSA_WORKLOAD_RECONSTRUCT:
        if (fft_len < 8192) return 1;
        if (fft_len < 32768) return 2;
        return 4;

    case SSA_WORKLOAD_WCORR:
        if (k < 10) return 1;
        if (k < 50) return 4;
        return max_threads;

    case SSA_WORKLOAD_FORECAST:
        return 1;

    default:
        return max_threads;
    }
}

static void mkl_config_ssa_decompose(int N, int L, int k)
{
    int threads = mkl_config_ssa_threads(N, L, k, SSA_WORKLOAD_DECOMPOSE);
    mkl_set_num_threads(threads);
}

static void mkl_config_ssa_reconstruct(int N, int n_group)
{
    int threads = mkl_config_ssa_threads(N, 0, n_group, SSA_WORKLOAD_RECONSTRUCT);
    mkl_set_num_threads(threads);
}

static void mkl_config_ssa_wcorr(int N, int k)
{
    int threads = mkl_config_ssa_threads(N, 0, k, SSA_WORKLOAD_WCORR);
    mkl_set_num_threads(threads);
}

/* ============================================================================
 * Memory Configuration
 * ============================================================================ */

#define SSA_MEM_ALIGN 64

static inline int mkl_config_is_aligned(const void* ptr)
{
    return ((uintptr_t)ptr & (SSA_MEM_ALIGN - 1)) == 0;
}

static size_t mkl_config_ssa_memory_estimate(int N, int L, int k)
{
    int K = N - L + 1;
    int fft_len = 1;
    while (fft_len < N) fft_len <<= 1;
    int r2c_len = fft_len / 2 + 1;

    size_t mem = 0;
    mem += L * k * sizeof(double);
    mem += K * k * sizeof(double);
    mem += k * sizeof(double) * 2;
    mem += N * sizeof(double);
    mem += 2 * r2c_len * sizeof(double);
    mem += fft_len * sizeof(double) * 2;
    mem += 2 * r2c_len * sizeof(double);
    mem += k * sizeof(double);
    mem += 32 * fft_len * sizeof(double);
    mem += 32 * 2 * r2c_len * sizeof(double);
    
    return mem;
}

static int mkl_config_fits_in_cache(int N, int L, int k)
{
    size_t mem = mkl_config_ssa_memory_estimate(N, L, k);
    size_t l3_bytes = (size_t)g_mkl_config.cpu.l3_cache_kb * 1024;
    return mem * 2 < l3_bytes;
}

/* ============================================================================
 * FFT Thread Tuning
 * ============================================================================ */

static int mkl_config_fft_threads(int fft_len)
{
    if (fft_len < 4096) return 1;
    if (fft_len < 16384) return 2;
    if (fft_len < 65536) return 4;
    return 8;
}

#define MKL_CONFIG_FFT_SET_THREADS(desc, fft_len) \
    DftiSetValue(desc, DFTI_THREAD_LIMIT, mkl_config_fft_threads(fft_len))

/* ============================================================================
 * Debug / Verbose Mode
 * ============================================================================ */

static void mkl_config_enable_verbose(void)
{
    MKL_SETENV("MKL_VERBOSE", "1");
}

static void mkl_config_print_status(void)
{
    printf("\n=== MKL Status ===\n");
    printf("Threads: %d\n", g_mkl_config.num_threads);
    printf("Mode: %s\n", g_mkl_config.opt_mode == MKL_LATENCY_MODE 
           ? "LATENCY" : "THROUGHPUT");
    printf("DAZ/FTZ: %s\n", g_mkl_config.daz_ftz_enabled ? "ON" : "OFF");
    printf("CPU: %s\n", g_mkl_config.cpu.cpu_name);
    printf("==================\n\n");
}

#endif /* SSA_USE_MKL */

#endif /* MKL_CONFIG_H */