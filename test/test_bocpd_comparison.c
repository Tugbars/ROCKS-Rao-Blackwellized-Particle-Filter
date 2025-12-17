/**
 * @file test_bocpd_comparison.c
 * @brief Targeted comparison: BOCPD-attached vs baseline RBPF
 *
 * Tests scenarios where BOCPD's "Afterburner" should provide value:
 *   1. Regime Starvation → Sudden Crisis
 *   2. Slow Drift (BOCPD shouldn't help - no regression check)
 *   3. Flash Crash + Recovery
 *   4. Double Changepoint
 *
 * Key metrics:
 *   - Transition lag (ticks to detect regime change)
 *   - Regime accuracy during transition windows
 *   - Min particles in target regime pre-switch
 *   - False positive rate
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rbpf_ksc.h"
#include "bocpd.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PORTABLE RNG (Windows doesn't have rand_r)
 *═══════════════════════════════════════════════════════════════════════════*/

static unsigned int rng_state;

static void rng_seed(unsigned int seed)
{
    rng_state = seed;
}

/* Simple xorshift32 - fast, portable, good enough for tests */
static unsigned int rng_next(void)
{
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static double rng_uniform(void)
{
    return (double)rng_next() / (double)0xFFFFFFFFU;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define N_PARTICLES 1000
#define N_REGIMES 3
#define SEED 42

/* Regime parameters: Calm (0), Normal (1), Crisis (2) */
static const double REGIME_MU_VOL[3] = {-5.0, -3.5, -2.0};
static const double REGIME_SIGMA_VOL[3] = {0.15, 0.20, 0.30};
static const double REGIME_PHI[3] = {0.98, 0.95, 0.90};

/* BOCPD configuration - tuned to reduce false positives */
#define BOCPD_LAMBDA 200.0 /* Expected run length (longer = less sensitive) */
#define BOCPD_MAX_RUN 512
#define BOCPD_Z_THRESHOLD 4.0  /* Z-score threshold (higher = less sensitive) */
#define BOCPD_DECAY 0.99       /* Slower decay for stability */
#define BOCPD_DELTA_WARMUP 100 /* Longer warmup for stable baseline */

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO DEFINITIONS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    const char *name;
    int n_ticks;
    int *true_regimes;    /* Ground truth regime per tick */
    double *observations; /* Synthetic returns */

    /* Transition points for analysis */
    int n_transitions;
    int *transition_ticks; /* Tick indices where regime changes */
    int *transition_from;  /* Source regime */
    int *transition_to;    /* Target regime */
} Scenario;

/* Generate observations from KSC model */
static void generate_ksc_observations(
    double *obs, double *true_h, int n_ticks,
    const int *regimes)
{
    double h = REGIME_MU_VOL[regimes[0]];

    for (int t = 0; t < n_ticks; t++)
    {
        int r = regimes[t];
        double mu = REGIME_MU_VOL[r];
        double sigma = REGIME_SIGMA_VOL[r];
        double phi = REGIME_PHI[r];

        /* State transition: h_t = mu + phi*(h_{t-1} - mu) + sigma*eta */
        double eta = (rng_uniform() - 0.5) * 3.46; /* ~N(0,1) approx */
        h = mu + phi * (h - mu) + sigma * eta;
        true_h[t] = h;

        /* Observation: r_t = exp(h_t/2) * epsilon */
        double eps = (rng_uniform() - 0.5) * 3.46;
        obs[t] = exp(h / 2.0) * eps;
    }
}

/*───────────────────────────────────────────────────────────────────────────
 * Scenario 1: Regime Starvation → Sudden Crisis
 *
 * 2000 ticks calm → instant jump to crisis
 * Tests: Can BOCPD resurrect a starved regime quickly?
 *───────────────────────────────────────────────────────────────────────────*/

static Scenario create_scenario_starvation(void)
{
    Scenario s;
    s.name = "Regime Starvation -> Crisis";
    s.n_ticks = 2500;
    s.true_regimes = (int *)malloc(s.n_ticks * sizeof(int));
    s.observations = (double *)malloc(s.n_ticks * sizeof(double));
    double *true_h = (double *)malloc(s.n_ticks * sizeof(double));

    /* 2000 ticks calm, then crisis */
    for (int t = 0; t < 2000; t++)
        s.true_regimes[t] = 0; /* Calm */
    for (int t = 2000; t < s.n_ticks; t++)
        s.true_regimes[t] = 2; /* Crisis */

    generate_ksc_observations(s.observations, true_h, s.n_ticks, s.true_regimes);
    free(true_h);

    /* One transition */
    s.n_transitions = 1;
    s.transition_ticks = (int *)malloc(sizeof(int));
    s.transition_from = (int *)malloc(sizeof(int));
    s.transition_to = (int *)malloc(sizeof(int));
    s.transition_ticks[0] = 2000;
    s.transition_from[0] = 0;
    s.transition_to[0] = 2;

    return s;
}

/*───────────────────────────────────────────────────────────────────────────
 * Scenario 2: Slow Drift (BOCPD shouldn't help)
 *
 * Gradual vol increase over 500 ticks
 * Tests: Confirm no regression when BOCPD doesn't fire
 *───────────────────────────────────────────────────────────────────────────*/

static Scenario create_scenario_slow_drift(void)
{
    Scenario s;
    s.name = "Slow Drift (no changepoint)";
    s.n_ticks = 1500;
    s.true_regimes = (int *)malloc(s.n_ticks * sizeof(int));
    s.observations = (double *)malloc(s.n_ticks * sizeof(double));
    double *true_h = (double *)malloc(s.n_ticks * sizeof(double));

    /* Start calm, drift through normal, end in crisis over 500 ticks */
    for (int t = 0; t < 500; t++)
        s.true_regimes[t] = 0; /* Calm */
    for (int t = 500; t < 1000; t++)
        s.true_regimes[t] = 1; /* Normal (drift zone) */
    for (int t = 1000; t < s.n_ticks; t++)
        s.true_regimes[t] = 2; /* Crisis */

    generate_ksc_observations(s.observations, true_h, s.n_ticks, s.true_regimes);
    free(true_h);

    /* Two gradual transitions */
    s.n_transitions = 2;
    s.transition_ticks = (int *)malloc(2 * sizeof(int));
    s.transition_from = (int *)malloc(2 * sizeof(int));
    s.transition_to = (int *)malloc(2 * sizeof(int));
    s.transition_ticks[0] = 500;
    s.transition_from[0] = 0;
    s.transition_to[0] = 1;
    s.transition_ticks[1] = 1000;
    s.transition_from[1] = 1;
    s.transition_to[1] = 2;

    return s;
}

/*───────────────────────────────────────────────────────────────────────────
 * Scenario 3: Flash Crash + Recovery
 *
 * 10-tick spike, then return to calm
 * Tests: BOCPD should fire twice (up and down), measure recovery lag
 *───────────────────────────────────────────────────────────────────────────*/

static Scenario create_scenario_flash_crash(void)
{
    Scenario s;
    s.name = "Flash Crash + Recovery";
    s.n_ticks = 1200;
    s.true_regimes = (int *)malloc(s.n_ticks * sizeof(int));
    s.observations = (double *)malloc(s.n_ticks * sizeof(double));
    double *true_h = (double *)malloc(s.n_ticks * sizeof(double));

    /* Calm → 20-tick crisis → calm */
    for (int t = 0; t < 500; t++)
        s.true_regimes[t] = 0; /* Calm */
    for (int t = 500; t < 520; t++)
        s.true_regimes[t] = 2; /* Flash crash */
    for (int t = 520; t < s.n_ticks; t++)
        s.true_regimes[t] = 0; /* Recovery */

    generate_ksc_observations(s.observations, true_h, s.n_ticks, s.true_regimes);
    free(true_h);

    /* Two transitions */
    s.n_transitions = 2;
    s.transition_ticks = (int *)malloc(2 * sizeof(int));
    s.transition_from = (int *)malloc(2 * sizeof(int));
    s.transition_to = (int *)malloc(2 * sizeof(int));
    s.transition_ticks[0] = 500;
    s.transition_from[0] = 0;
    s.transition_to[0] = 2;
    s.transition_ticks[1] = 520;
    s.transition_from[1] = 2;
    s.transition_to[1] = 0;

    return s;
}

/*───────────────────────────────────────────────────────────────────────────
 * Scenario 4: Double Changepoint
 *
 * Crisis → Brief calm (50 ticks) → Crisis again
 * Tests: BOCPD cooldown/dwell time handling
 *───────────────────────────────────────────────────────────────────────────*/

static Scenario create_scenario_double_changepoint(void)
{
    Scenario s;
    s.name = "Double Changepoint";
    s.n_ticks = 1500;
    s.true_regimes = (int *)malloc(s.n_ticks * sizeof(int));
    s.observations = (double *)malloc(s.n_ticks * sizeof(double));
    double *true_h = (double *)malloc(s.n_ticks * sizeof(double));

    /* Calm → Crisis → Brief calm → Crisis → Calm */
    for (int t = 0; t < 400; t++)
        s.true_regimes[t] = 0; /* Calm */
    for (int t = 400; t < 700; t++)
        s.true_regimes[t] = 2; /* Crisis 1 */
    for (int t = 700; t < 750; t++)
        s.true_regimes[t] = 0; /* Brief calm */
    for (int t = 750; t < 1100; t++)
        s.true_regimes[t] = 2; /* Crisis 2 */
    for (int t = 1100; t < s.n_ticks; t++)
        s.true_regimes[t] = 0; /* Final calm */

    generate_ksc_observations(s.observations, true_h, s.n_ticks, s.true_regimes);
    free(true_h);

    /* Four transitions */
    s.n_transitions = 4;
    s.transition_ticks = (int *)malloc(4 * sizeof(int));
    s.transition_from = (int *)malloc(4 * sizeof(int));
    s.transition_to = (int *)malloc(4 * sizeof(int));
    s.transition_ticks[0] = 400;
    s.transition_from[0] = 0;
    s.transition_to[0] = 2;
    s.transition_ticks[1] = 700;
    s.transition_from[1] = 2;
    s.transition_to[1] = 0;
    s.transition_ticks[2] = 750;
    s.transition_from[2] = 0;
    s.transition_to[2] = 2;
    s.transition_ticks[3] = 1100;
    s.transition_from[3] = 2;
    s.transition_to[3] = 0;

    return s;
}

static void free_scenario(Scenario *s)
{
    free(s->true_regimes);
    free(s->observations);
    free(s->transition_ticks);
    free(s->transition_from);
    free(s->transition_to);
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    /* Per-transition metrics */
    double avg_transition_lag; /* Ticks to detect regime change */
    double max_transition_lag;

    /* Accuracy in transition windows (±50 ticks around changepoint) */
    double transition_window_accuracy;

    /* Overall accuracy */
    double overall_accuracy;

    /* BOCPD-specific */
    int bocpd_triggers;  /* Number of times BOCPD fired */
    int true_positives;  /* BOCPD fired at actual changepoint (±10 ticks) */
    int false_positives; /* BOCPD fired with no nearby changepoint */

    /* Particle health */
    double avg_ess;
    double min_ess;
    double min_particles_at_transition; /* Min particles in target regime when transition happened */
} Metrics;

static int detect_regime(const RBPF_KSC_Output *out)
{
    /* Use smoothed_regime from SPRT */
    return out->smoothed_regime;
}

static int is_near_transition(int tick, const Scenario *s, int window)
{
    for (int i = 0; i < s->n_transitions; i++)
    {
        if (abs(tick - s->transition_ticks[i]) <= window)
            return 1;
    }
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN SINGLE CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

static Metrics run_rbpf(
    const Scenario *s,
    int use_bocpd,
    bocpd_t *bocpd,
    bocpd_delta_detector_t *delta,
    bocpd_hazard_t *hazard)
{
    Metrics m = {0};

    /* Create RBPF */
    RBPF_KSC *rbpf = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
    if (!rbpf)
    {
        fprintf(stderr, "Failed to create RBPF\n");
        return m;
    }

    /* Configure regimes */
    for (int r = 0; r < N_REGIMES; r++)
    {
        rbpf_ksc_set_regime_params(rbpf, r,
                                   (rbpf_real_t)REGIME_MU_VOL[r],
                                   (rbpf_real_t)REGIME_SIGMA_VOL[r],
                                   (rbpf_real_t)REGIME_PHI[r]);
    }

    /* Attach BOCPD if enabled */
    if (use_bocpd && bocpd && delta)
    {
        bocpd_reset(bocpd);
        bocpd_delta_init(delta, BOCPD_DELTA_WARMUP);
        rbpf_ksc_attach_bocpd(rbpf, bocpd, delta, hazard);
        rbpf_ksc_set_bocpd_params(rbpf, BOCPD_Z_THRESHOLD, BOCPD_DECAY, 1000);
    }

    /* Initialize */
    rbpf_ksc_init(rbpf, (rbpf_real_t)REGIME_MU_VOL[0], SEED);

    /* Tracking */
    int correct = 0;
    int correct_transition = 0;
    int transition_window_ticks = 0;
    double ess_sum = 0.0;
    double min_ess = 1e30;

    /* Transition lag tracking */
    int *detected_at = (int *)calloc(s->n_transitions, sizeof(int));
    for (int i = 0; i < s->n_transitions; i++)
        detected_at[i] = -1;

    /* Run filter */
    RBPF_KSC_Output out;

    for (int t = 0; t < s->n_ticks; t++)
    {
        rbpf_ksc_step(rbpf, (rbpf_real_t)s->observations[t], &out);

        int detected = detect_regime(&out);
        int truth = s->true_regimes[t];

        /* Overall accuracy */
        if (detected == truth)
            correct++;

        /* Transition window accuracy */
        if (is_near_transition(t, s, 50))
        {
            transition_window_ticks++;
            if (detected == truth)
                correct_transition++;
        }

        /* ESS tracking */
        ess_sum += out.ess;
        if (out.ess < min_ess)
            min_ess = out.ess;

        /* Transition lag: when did we first detect the new regime? */
        for (int i = 0; i < s->n_transitions; i++)
        {
            if (t >= s->transition_ticks[i] && detected_at[i] < 0)
            {
                if (detected == s->transition_to[i])
                {
                    detected_at[i] = t - s->transition_ticks[i];
                }
            }
        }

        /* BOCPD tracking */
        if (use_bocpd && out.bocpd_triggered)
        {
            m.bocpd_triggers++;
            if (is_near_transition(t, s, 10))
            {
                m.true_positives++;
            }
            else
            {
                m.false_positives++;
            }
        }
    }

    /* Compute metrics */
    m.overall_accuracy = (double)correct / s->n_ticks;
    m.transition_window_accuracy = (transition_window_ticks > 0)
                                       ? (double)correct_transition / transition_window_ticks
                                       : 0.0;

    m.avg_ess = ess_sum / s->n_ticks;
    m.min_ess = min_ess;

    /* Transition lag */
    double lag_sum = 0.0;
    double max_lag = 0.0;
    int detected_count = 0;

    for (int i = 0; i < s->n_transitions; i++)
    {
        if (detected_at[i] >= 0)
        {
            lag_sum += detected_at[i];
            if (detected_at[i] > max_lag)
                max_lag = detected_at[i];
            detected_count++;
        }
        else
        {
            /* Never detected - penalize with scenario length */
            lag_sum += (s->n_ticks - s->transition_ticks[i]);
            max_lag = s->n_ticks;
        }
    }

    m.avg_transition_lag = (s->n_transitions > 0) ? lag_sum / s->n_transitions : 0.0;
    m.max_transition_lag = max_lag;

    free(detected_at);
    rbpf_ksc_destroy(rbpf);

    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_comparison(const char *name, Metrics baseline, Metrics bocpd_m)
{
    printf("\n┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-63s │\n", name);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ Metric                          Baseline      BOCPD      Delta │\n");
    printf("├─────────────────────────────────────────────────────────────────┤\n");

    printf("│ Avg Transition Lag (ticks)      %7.1f     %7.1f    %+6.1f │\n",
           baseline.avg_transition_lag, bocpd_m.avg_transition_lag,
           bocpd_m.avg_transition_lag - baseline.avg_transition_lag);

    printf("│ Max Transition Lag (ticks)      %7.1f     %7.1f    %+6.1f │\n",
           baseline.max_transition_lag, bocpd_m.max_transition_lag,
           bocpd_m.max_transition_lag - baseline.max_transition_lag);

    printf("│ Transition Window Accuracy      %6.1f%%     %6.1f%%   %+5.1f%% │\n",
           baseline.transition_window_accuracy * 100,
           bocpd_m.transition_window_accuracy * 100,
           (bocpd_m.transition_window_accuracy - baseline.transition_window_accuracy) * 100);

    printf("│ Overall Accuracy                %6.1f%%     %6.1f%%   %+5.1f%% │\n",
           baseline.overall_accuracy * 100,
           bocpd_m.overall_accuracy * 100,
           (bocpd_m.overall_accuracy - baseline.overall_accuracy) * 100);

    printf("│ Avg ESS                         %7.1f     %7.1f    %+6.1f │\n",
           baseline.avg_ess, bocpd_m.avg_ess,
           bocpd_m.avg_ess - baseline.avg_ess);

    printf("│ Min ESS                         %7.1f     %7.1f    %+6.1f │\n",
           baseline.min_ess, bocpd_m.min_ess,
           bocpd_m.min_ess - baseline.min_ess);

    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ BOCPD Triggers                      N/A     %7d            │\n",
           bocpd_m.bocpd_triggers);
    printf("│ True Positives (±10 ticks)          N/A     %7d            │\n",
           bocpd_m.true_positives);
    printf("│ False Positives                     N/A     %7d            │\n",
           bocpd_m.false_positives);
    printf("└─────────────────────────────────────────────────────────────────┘\n");
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  BOCPD vs Baseline RBPF Comparison Test\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Particles: %d\n", N_PARTICLES);
    printf("  Regimes: %d (Calm/Normal/Crisis)\n", N_REGIMES);
    printf("  BOCPD z-threshold: %.1f\n", BOCPD_Z_THRESHOLD);
    printf("  Pilot light: 2 particles, 0.1%% mutation\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Initialize BOCPD (shared across scenarios) */
    bocpd_hazard_t hazard;
    bocpd_t bocpd;
    bocpd_delta_detector_t delta;

    bocpd_prior_t prior = {
        .mu0 = -5.0,   /* Expected mean log-vol */
        .kappa0 = 1.0, /* Low confidence in mean */
        .alpha0 = 3.0, /* Shape parameter */
        .beta0 = 15.0  /* Mean variance = beta/(alpha-1) ≈ 7.5 (covers KSC noise ~5.0) */
    };

    if (bocpd_hazard_init_constant(&hazard, BOCPD_LAMBDA, BOCPD_MAX_RUN) != 0)
    {
        fprintf(stderr, "Failed to init hazard\n");
        return 1;
    }

    if (bocpd_init_with_hazard(&bocpd, &hazard, prior) != 0)
    {
        fprintf(stderr, "Failed to init BOCPD\n");
        return 1;
    }

    bocpd_delta_init(&delta, BOCPD_DELTA_WARMUP);

    /* Initialize RNG */
    rng_seed(SEED);

    /* Scenario 1: Regime Starvation */
    {
        Scenario s = create_scenario_starvation();
        printf("\nRunning: %s (%d ticks, %d transitions)...\n",
               s.name, s.n_ticks, s.n_transitions);

        Metrics baseline = run_rbpf(&s, 0, NULL, NULL, NULL);
        Metrics bocpd_m = run_rbpf(&s, 1, &bocpd, &delta, &hazard);

        print_comparison(s.name, baseline, bocpd_m);
        free_scenario(&s);
    }

    /* Scenario 2: Slow Drift */
    {
        Scenario s = create_scenario_slow_drift();
        printf("\nRunning: %s (%d ticks, %d transitions)...\n",
               s.name, s.n_ticks, s.n_transitions);

        Metrics baseline = run_rbpf(&s, 0, NULL, NULL, NULL);
        Metrics bocpd_m = run_rbpf(&s, 1, &bocpd, &delta, &hazard);

        print_comparison(s.name, baseline, bocpd_m);
        free_scenario(&s);
    }

    /* Scenario 3: Flash Crash */
    {
        Scenario s = create_scenario_flash_crash();
        printf("\nRunning: %s (%d ticks, %d transitions)...\n",
               s.name, s.n_ticks, s.n_transitions);

        Metrics baseline = run_rbpf(&s, 0, NULL, NULL, NULL);
        Metrics bocpd_m = run_rbpf(&s, 1, &bocpd, &delta, &hazard);

        print_comparison(s.name, baseline, bocpd_m);
        free_scenario(&s);
    }

    /* Scenario 4: Double Changepoint */
    {
        Scenario s = create_scenario_double_changepoint();
        printf("\nRunning: %s (%d ticks, %d transitions)...\n",
               s.name, s.n_ticks, s.n_transitions);

        Metrics baseline = run_rbpf(&s, 0, NULL, NULL, NULL);
        Metrics bocpd_m = run_rbpf(&s, 1, &bocpd, &delta, &hazard);

        print_comparison(s.name, baseline, bocpd_m);
        free_scenario(&s);
    }

    /* Cleanup */
    bocpd_free(&bocpd);
    bocpd_hazard_free(&hazard);

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  Test complete.\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}