/**
 * @file bocpd.h
 * @brief Bayesian Online Changepoint Detection for MMPF integration
 * 
 * Streamlined implementation with:
 * - Power-law hazard H(r) = α/(r+1) for heavy-tailed regime durations
 * - Student-t predictive with NIG conjugate prior
 * - AVX2 SIMD optimization
 * - Delta detector for MMPF shock triggering
 */

#ifndef BOCPD_H
#define BOCPD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * INTERLEAVED BLOCK LAYOUT (256 bytes per 4 run lengths)
 *═══════════════════════════════════════════════════════════════════════════*/

#define BOCPD_IBLK_MU        0
#define BOCPD_IBLK_C1       32
#define BOCPD_IBLK_C2       64
#define BOCPD_IBLK_INV_SSN  96
#define BOCPD_IBLK_KAPPA   128
#define BOCPD_IBLK_ALPHA   160
#define BOCPD_IBLK_BETA    192
#define BOCPD_IBLK_SS_N    224
#define BOCPD_IBLK_STRIDE  256
#define BOCPD_IBLK_DOUBLES  32

#define BOCPD_CUR_BUF(b)  ((b)->interleaved[(b)->cur_buf])
#define BOCPD_NEXT_BUF(b) ((b)->interleaved[1 - (b)->cur_buf])

/*═══════════════════════════════════════════════════════════════════════════
 * HAZARD FUNCTION TYPES
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    HAZARD_CONSTANT,     /* H = 1/λ (geometric durations) */
    HAZARD_POWER_LAW,    /* H(r) = α/(r+1) (heavy-tailed durations) */
    HAZARD_LEARNED       /* Beta-Bernoulli online learning */
} bocpd_hazard_type_t;

typedef struct {
    bocpd_hazard_type_t type;
    size_t max_run_length;
    
    /* Precomputed hazard table */
    double *h;           /* H(r) for r = 0..max_run_length-1 */
    double *one_minus_h; /* 1 - H(r) */
    
    /* Type-specific parameters */
    union {
        struct { double lambda; } constant;
        struct { double alpha; } power_law;
        struct { double a, b; size_t n_obs, n_cp; } learned;
    } params;
    
} bocpd_hazard_t;

/*═══════════════════════════════════════════════════════════════════════════
 * PRIOR PARAMETERS (Normal-Inverse-Gamma)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double mu0;     /* Prior mean */
    double kappa0;  /* Mean pseudo-count (confidence) */
    double alpha0;  /* Variance shape (typically 1-2) */
    double beta0;   /* Variance rate */
} bocpd_prior_t;

/*═══════════════════════════════════════════════════════════════════════════
 * DELTA DETECTOR (Storvik self-calibrating threshold)
 *═══════════════════════════════════════════════════════════════════════════*/

#define BOCPD_SHORT_RUN_WINDOW 15

typedef struct {
    /* Sufficient statistics for delta ~ N(μ, σ²) */
    double kappa;       /* Pseudo-count for mean */
    double mu;          /* Running mean of delta */
    double alpha;       /* Shape for variance */
    double beta;        /* Rate for variance */
    
    /* Previous state */
    double prev_short_mass;
    
    /* Warmup tracking */
    size_t n_observations;
    size_t warmup_period;
    
} bocpd_delta_detector_t;

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN DETECTOR STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Configuration */
    size_t capacity;
    double trunc_thresh;
    bocpd_prior_t prior;
    
    /* Hazard (can be constant or run-length dependent) */
    double hazard;        /* Constant hazard (if not using hazard table) */
    double one_minus_h;   /* 1 - hazard */
    bocpd_hazard_t *hazard_table;  /* NULL = use constant hazard */
    
    /* Precomputed prior lgamma values */
    double prior_lgamma_alpha;
    double prior_lgamma_alpha_p5;
    
    /* Ping-pong buffers for posteriors */
    double *interleaved[2];
    int cur_buf;
    
    /* Run-length probability distribution */
    double *r;
    double *r_scratch;
    size_t active_len;
    
    /* Outputs */
    size_t t;
    size_t map_runlength;
    double p_changepoint;
    
    /* Memory management */
    void *mega;
    size_t mega_bytes;
    
} bocpd_t;

/*═══════════════════════════════════════════════════════════════════════════
 * HAZARD FUNCTION API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize constant hazard: H = 1/λ
 */
int bocpd_hazard_init_constant(bocpd_hazard_t *h, double lambda, size_t max_run);

/**
 * Initialize power-law hazard: H(r) = α/(r+1)
 * Recommended for finance: α ∈ [0.5, 1.0]
 */
int bocpd_hazard_init_power_law(bocpd_hazard_t *h, double alpha, size_t max_run);

/**
 * Initialize learned hazard with Beta(a, b) prior
 */
int bocpd_hazard_init_learned(bocpd_hazard_t *h, double a, double b, size_t max_run);

/**
 * Update learned hazard after observing changepoint/non-changepoint
 */
void bocpd_hazard_learn_update(bocpd_hazard_t *h, int is_changepoint, size_t decay_window);

/**
 * Free hazard table memory
 */
void bocpd_hazard_free(bocpd_hazard_t *h);

/*═══════════════════════════════════════════════════════════════════════════
 * DELTA DETECTOR API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize delta detector with Storvik learning
 */
void bocpd_delta_init(bocpd_delta_detector_t *d, size_t warmup);

/**
 * Update detector with current run-length distribution
 * Returns the current delta value
 */
double bocpd_delta_update(bocpd_delta_detector_t *d, const double *r, 
                          size_t active_len, double decay);

/**
 * Check if changepoint detected (z-score > threshold)
 */
int bocpd_delta_check(const bocpd_delta_detector_t *d, double z_threshold);

/**
 * Get current z-score
 */
double bocpd_delta_zscore(const bocpd_delta_detector_t *d, double delta);

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN DETECTOR API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize detector with constant hazard
 */
int bocpd_init(bocpd_t *b, double hazard_lambda, bocpd_prior_t prior, 
               size_t max_run_length);

/**
 * Initialize detector with custom hazard function
 */
int bocpd_init_with_hazard(bocpd_t *b, bocpd_hazard_t *hazard, 
                           bocpd_prior_t prior);

/**
 * Free detector memory
 */
void bocpd_free(bocpd_t *b);

/**
 * Reset to initial state (no reallocation)
 */
void bocpd_reset(bocpd_t *b);

/**
 * Process single observation
 * Updates: p_changepoint, map_runlength, r[]
 */
void bocpd_step(bocpd_t *b, double x);

/**
 * Get short-run mass P(r < window)
 */
double bocpd_short_mass(const bocpd_t *b, size_t window);

/*═══════════════════════════════════════════════════════════════════════════
 * MMPF INTEGRATION
 * 
 * Usage:
 *   bocpd_t bocpd;
 *   bocpd_hazard_t hazard;
 *   bocpd_delta_detector_t delta;
 *   
 *   bocpd_hazard_init_power_law(&hazard, 0.8, 1024);
 *   bocpd_init_with_hazard(&bocpd, &hazard, prior);
 *   bocpd_delta_init(&delta, 100);
 *   
 *   for each observation x:
 *       bocpd_step(&bocpd, x);
 *       bocpd_delta_update(&delta, bocpd.r, bocpd.active_len, 0.995);
 *       
 *       if (bocpd_delta_check(&delta, 3.0)) {
 *           mmpf_inject_shock(mmpf);
 *           mmpf_step(mmpf, x, &out);
 *           mmpf_restore_from_shock(mmpf);
 *       } else {
 *           mmpf_step(mmpf, x, &out);
 *       }
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
}
#endif

#endif /* BOCPD_H */
