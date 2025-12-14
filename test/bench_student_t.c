/**
 * @file bench_student_t_mkl.c
 * @brief Benchmark: Scalar vs MKL Student-t update performance
 *
 * Cross-platform (Windows + Linux)
 */

#include "rbpf_ksc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/*---------------------------------------------------------------------------*/

static void run_benchmark(int n_particles, int n_steps, int n_warmup)
{
    RBPF_KSC *rbpf;
    RBPF_RobustOCSN ocsn;
    double t0, t1;
    double scalar_time, mkl_time;
    double scalar_us, mkl_us;
    rbpf_real_t nu = RBPF_REAL(5.0);
    int i;

    /* Create filter */
    rbpf = rbpf_ksc_create(n_particles, 2);
    if (!rbpf)
    {
        printf("Failed to create RBPF with %d particles\n", n_particles);
        return;
    }

    /* Initialize particles */
    rbpf_ksc_init(rbpf, RBPF_REAL(-4.0), RBPF_REAL(1.0));

    /* Enable Student-t */
    rbpf_ksc_enable_student_t(rbpf, nu);

    /* Setup OCSN */
    ocsn.enabled = 1;
    ocsn.regime[0].prob = RBPF_REAL(0.01);
    ocsn.regime[0].variance = RBPF_REAL(100.0);
    ocsn.regime[1].prob = RBPF_REAL(0.01);
    ocsn.regime[1].variance = RBPF_REAL(100.0);

    /* Warmup - scalar */
    for (i = 0; i < n_warmup; i++)
    {
        rbpf_real_t y = RBPF_REAL(0.01) * (rbpf_real_t)(i % 100 - 50);
        rbpf_ksc_predict(rbpf);
        rbpf_ksc_update_student_t_robust(rbpf, y, nu, &ocsn);
    }

    /* Reset state */
    rbpf_ksc_destroy(rbpf);
    rbpf = rbpf_ksc_create(n_particles, 2);
    rbpf_ksc_init(rbpf, RBPF_REAL(-4.0), RBPF_REAL(1.0));
    rbpf_ksc_enable_student_t(rbpf, nu);

    /* Benchmark scalar */
    t0 = get_time_sec();
    for (i = 0; i < n_steps; i++)
    {
        rbpf_real_t y = RBPF_REAL(0.01) * (rbpf_real_t)(i % 100 - 50);
        rbpf_ksc_predict(rbpf);
        rbpf_ksc_update_student_t_robust(rbpf, y, nu, &ocsn);
    }
    t1 = get_time_sec();
    scalar_time = t1 - t0;

    /* Reset state */
    rbpf_ksc_destroy(rbpf);
    rbpf = rbpf_ksc_create(n_particles, 2);
    rbpf_ksc_init(rbpf, RBPF_REAL(-4.0), RBPF_REAL(1.0));
    rbpf_ksc_enable_student_t(rbpf, nu);

    /* Warmup - MKL */
    for (i = 0; i < n_warmup; i++)
    {
        rbpf_real_t y = RBPF_REAL(0.01) * (rbpf_real_t)(i % 100 - 50);
        rbpf_ksc_predict(rbpf);
        rbpf_ksc_update_student_t_robust_mkl(rbpf, y, nu, &ocsn);
    }

    /* Reset state */
    rbpf_ksc_destroy(rbpf);
    rbpf = rbpf_ksc_create(n_particles, 2);
    rbpf_ksc_init(rbpf, RBPF_REAL(-4.0), RBPF_REAL(1.0));
    rbpf_ksc_enable_student_t(rbpf, nu);

    /* Benchmark MKL */
    t0 = get_time_sec();
    for (i = 0; i < n_steps; i++)
    {
        rbpf_real_t y = RBPF_REAL(0.01) * (rbpf_real_t)(i % 100 - 50);
        rbpf_ksc_predict(rbpf);
        rbpf_ksc_update_student_t_robust_mkl(rbpf, y, nu, &ocsn);
    }
    t1 = get_time_sec();
    mkl_time = t1 - t0;

    /* Results */
    scalar_us = (scalar_time / n_steps) * 1e6;
    mkl_us = (mkl_time / n_steps) * 1e6;

    printf("  n=%5d | Scalar: %8.2f us | MKL: %8.2f us | Speedup: %.2fx\n",
           n_particles, scalar_us, mkl_us, scalar_us / mkl_us);

    rbpf_ksc_destroy(rbpf);
}

/*---------------------------------------------------------------------------*/

int main(void)
{
    printf("=== Student-t Robust Update Benchmark ===\n");
    printf("  (Scalar vs MKL-optimized)\n\n");

    printf("Configuration:\n");
#if RBPF_USE_DOUBLE
    printf("  Precision: double\n");
#else
    printf("  Precision: float\n");
#endif
    printf("  MKL block size: %d\n\n", 2048);

    printf("Results (per update step):\n");
    printf("─────────────────────────────────────────────────────────────────\n");

    run_benchmark(100, 5000, 500);
    run_benchmark(500, 2000, 200);
    run_benchmark(1000, 1000, 100);
    run_benchmark(2000, 500, 50);
    run_benchmark(5000, 200, 20);
    run_benchmark(10000, 100, 10);

    printf("─────────────────────────────────────────────────────────────────\n");
    printf("\nNote: Speedup increases with particle count due to MKL batching.\n");
    printf("      At low n, scalar may win due to MKL overhead.\n");

    return 0;
}