/**
 * @file example_usage.c
 * @brief Practical Example: Integrating Oracle Bridge V2 with RBPF
 *
 * This example demonstrates the complete workflow:
 * 1. RBPF runs every tick, pushes observations
 * 2. Quality monitoring detects Π degradation EARLY
 * 3. Oracle (simulated) provides corrected Π
 * 4. Injection decision considers urgency + confidence
 * 5. Blending applies the correction
 *
 * The key insight: We monitor PREDICTION QUALITY, not just ESS.
 * By the time ESS drops, it's too late.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "oracle_bridge_v2.h"

/*═══════════════════════════════════════════════════════════════════════════
 * SIMULATED REGIME-SWITCHING PROCESS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int K;                    /* Number of regimes */
    int current_regime;       /* Current state */
    double *means;            /* Regime means */
    double *volatilities;     /* Regime volatilities */
    float *true_Pi;           /* True transition matrix */
    uint64_t rng_state;
} MarketSimulator;

static double sim_uniform(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) * 5.421010862427522e-20;
}

static double sim_normal(uint64_t *state) {
    double u1 = sim_uniform(state);
    double u2 = sim_uniform(state);
    while (u1 < 1e-10) u1 = sim_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
}

static void sim_init(MarketSimulator *sim, int K, uint64_t seed) {
    sim->K = K;
    sim->current_regime = 0;
    sim->rng_state = seed;
    
    sim->means = malloc(K * sizeof(double));
    sim->volatilities = malloc(K * sizeof(double));
    sim->true_Pi = malloc(K * K * sizeof(float));
    
    /* Regime parameters */
    sim->means[0] = 0.0;   sim->volatilities[0] = 0.01;  /* Calm */
    sim->means[1] = 0.02;  sim->volatilities[1] = 0.03;  /* Bull */
    sim->means[2] = -0.02; sim->volatilities[2] = 0.05;  /* Bear */
    
    /* True transition matrix */
    float Pi[9] = {
        0.95f, 0.03f, 0.02f,   /* Calm: stays calm */
        0.05f, 0.90f, 0.05f,   /* Bull: moderate persistence */
        0.05f, 0.05f, 0.90f    /* Bear: moderate persistence */
    };
    memcpy(sim->true_Pi, Pi, 9 * sizeof(float));
}

static double sim_step(MarketSimulator *sim) {
    /* Transition */
    double u = sim_uniform(&sim->rng_state);
    float cumsum = 0;
    int new_regime = sim->current_regime;
    
    for (int j = 0; j < sim->K; j++) {
        cumsum += sim->true_Pi[sim->current_regime * sim->K + j];
        if (u < cumsum) {
            new_regime = j;
            break;
        }
    }
    sim->current_regime = new_regime;
    
    /* Generate observation */
    double mean = sim->means[new_regime];
    double vol = sim->volatilities[new_regime];
    return mean + vol * sim_normal(&sim->rng_state);
}

static void sim_free(MarketSimulator *sim) {
    free(sim->means);
    free(sim->volatilities);
    free(sim->true_Pi);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SIMULATED RBPF (Simplified)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int K;
    int N;                    /* Number of particles */
    double *weights;          /* Particle weights */
    int *regimes;             /* Particle regime assignments */
    float *Pi;                /* Current operating Π */
    float *Q_storvik;         /* Storvik counts */
    double prediction;        /* Last prediction */
    double log_likelihood;    /* Last log-likelihood */
    int map_regime;           /* MAP estimate */
} SimpleRBPF;

static void rbpf_init(SimpleRBPF *rbpf, int K, int N, const float *Pi_init) {
    rbpf->K = K;
    rbpf->N = N;
    
    rbpf->weights = malloc(N * sizeof(double));
    rbpf->regimes = malloc(N * sizeof(int));
    rbpf->Pi = malloc(K * K * sizeof(float));
    rbpf->Q_storvik = malloc(K * K * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        rbpf->weights[i] = 1.0 / N;
        rbpf->regimes[i] = 0;
    }
    
    memcpy(rbpf->Pi, Pi_init, K * K * sizeof(float));
    
    /* Initialize Storvik counts */
    for (int i = 0; i < K * K; i++) {
        rbpf->Q_storvik[i] = Pi_init[i] * 100.0f;
    }
}

static void rbpf_update(SimpleRBPF *rbpf, double observation, 
                        const double *regime_means, const double *regime_vols) {
    int K = rbpf->K;
    int N = rbpf->N;
    
    /* Compute prediction (weighted average) */
    rbpf->prediction = 0;
    for (int i = 0; i < N; i++) {
        rbpf->prediction += rbpf->weights[i] * regime_means[rbpf->regimes[i]];
    }
    
    /* Update weights based on likelihood */
    double total_weight = 0;
    double log_lik = 0;
    
    for (int i = 0; i < N; i++) {
        int z = rbpf->regimes[i];
        double mean = regime_means[z];
        double vol = regime_vols[z];
        
        double error = observation - mean;
        double ll = -0.5 * (error * error) / (vol * vol) - log(vol);
        
        rbpf->weights[i] *= exp(ll);
        total_weight += rbpf->weights[i];
        log_lik += ll;
    }
    
    rbpf->log_likelihood = log_lik / N;
    
    /* Normalize */
    for (int i = 0; i < N; i++) {
        rbpf->weights[i] /= total_weight;
    }
    
    /* Find MAP regime */
    double regime_probs[8] = {0};
    for (int i = 0; i < N; i++) {
        regime_probs[rbpf->regimes[i]] += rbpf->weights[i];
    }
    rbpf->map_regime = 0;
    for (int k = 1; k < K; k++) {
        if (regime_probs[k] > regime_probs[rbpf->map_regime]) {
            rbpf->map_regime = k;
        }
    }
    
    /* Resample and transition (simplified) */
    int new_regimes[1024];
    for (int i = 0; i < N; i++) {
        /* Multinomial resample */
        double u = (double)rand() / RAND_MAX;
        double cumsum = 0;
        int parent = 0;
        for (int j = 0; j < N; j++) {
            cumsum += rbpf->weights[j];
            if (u < cumsum) { parent = j; break; }
        }
        
        /* Transition */
        int old_z = rbpf->regimes[parent];
        u = (double)rand() / RAND_MAX;
        cumsum = 0;
        int new_z = old_z;
        for (int k = 0; k < K; k++) {
            cumsum += rbpf->Pi[old_z * K + k];
            if (u < cumsum) { new_z = k; break; }
        }
        new_regimes[i] = new_z;
        
        /* Update Storvik counts */
        rbpf->Q_storvik[old_z * K + new_z] += 0.01f;
    }
    
    memcpy(rbpf->regimes, new_regimes, N * sizeof(int));
    
    /* Reset weights */
    for (int i = 0; i < N; i++) {
        rbpf->weights[i] = 1.0 / N;
    }
}

static void rbpf_set_pi(SimpleRBPF *rbpf, const float *Pi, const float *Q) {
    memcpy(rbpf->Pi, Pi, rbpf->K * rbpf->K * sizeof(float));
    if (Q) {
        memcpy(rbpf->Q_storvik, Q, rbpf->K * rbpf->K * sizeof(float));
    }
}

static void rbpf_free(SimpleRBPF *rbpf) {
    free(rbpf->weights);
    free(rbpf->regimes);
    free(rbpf->Pi);
    free(rbpf->Q_storvik);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN SIMULATION
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    srand(time(NULL));
    
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║        ORACLE BRIDGE V2 - INTEGRATION EXAMPLE                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Demonstrating EARLY detection of Π degradation                   ║\n");
    printf("║  via prediction quality monitoring                                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    int K = 3;
    int N = 100;  /* Particles */
    int T = 2000; /* Ticks */
    
    /*═══════════════════════════════════════════════════════════════════
     * SETUP
     *═══════════════════════════════════════════════════════════════════*/
    
    /* Market simulator with TRUE Π */
    MarketSimulator market;
    sim_init(&market, K, 12345);
    
    /* RBPF with WRONG initial Π (deliberately bad) */
    float wrong_Pi[9] = {
        0.80f, 0.10f, 0.10f,   /* Less sticky than true */
        0.10f, 0.80f, 0.10f,
        0.10f, 0.10f, 0.80f
    };
    
    SimpleRBPF rbpf;
    rbpf_init(&rbpf, K, N, wrong_Pi);
    
    /* Observation buffer (shared) */
    ObservationBuffer obs_buffer;
    obs_buffer_init(&obs_buffer);
    
    /* Oracle Bridge */
    OracleBridgeV2 bridge;
    OracleBridgeConfig config = oracle_bridge_config_defaults(K);
    config.verbose = false;
    
    oracle_bridge_init(&bridge, &config, &obs_buffer, wrong_Pi);
    
    /*═══════════════════════════════════════════════════════════════════
     * SIMULATION LOOP
     *═══════════════════════════════════════════════════════════════════*/
    
    printf("Running %d ticks...\n\n", T);
    printf("Legend:\n");
    printf("  RMSE_r = RMSE ratio (>1.5 = degraded predictions)\n");
    printf("  D3     = Self-contradiction (Storvik vs Operating)\n");
    printf("  ESS    = Effective Sample Size ratio\n");
    printf("  Q      = Quality score (0-1)\n\n");
    
    int injection_count = 0;
    int oracle_submit_interval = 200;  /* Oracle runs slower */
    
    for (int t = 0; t < T; t++) {
        /* Generate observation */
        double obs = sim_step(&market);
        
        /* Push to buffer */
        obs_buffer_push(&obs_buffer, obs, t);
        
        /* RBPF update */
        rbpf_update(&rbpf, obs, market.means, market.volatilities);
        
        /* Update bridge Storvik */
        oracle_bridge_update_storvik(&bridge, rbpf.Q_storvik);
        
        /* Bridge tick - monitors quality and checks for Oracle */
        InjectionDecision dec = oracle_bridge_rbpf_tick(
            &bridge,
            obs,
            rbpf.prediction,
            rbpf.log_likelihood,
            rbpf.map_regime,
            rbpf.weights,
            N,
            t);
        
        /* Periodic Oracle submission (simulating async Oracle thread) */
        if (t > 0 && t % oracle_submit_interval == 0) {
            /* Oracle "discovers" the true Π */
            oracle_bridge_submit_oracle(
                &bridge,
                market.true_Pi,       /* True Π */
                NULL,                 /* No Q */
                0.18f,                /* Good acceptance rate */
                150.0f,               /* Good count */
                50,                   /* Sweeps */
                t - oracle_submit_interval,
                t);
        }
        
        /* Handle injection */
        if (dec.should_inject) {
            PiQualitySnapshot q = oracle_bridge_get_quality(&bridge);
            
            printf("[t=%4d] INJECTION TRIGGERED\n", t);
            printf("         Urgency: %s (score=%.2f)\n", 
                   injection_urgency_str(dec.urgency), dec.urgency_score);
            printf("         Source:  %s\n", injection_source_str(dec.source));
            printf("         Gamma:   %.3f\n", dec.gamma);
            printf("         Reason:  %s\n", dec.reason);
            printf("         Quality: RMSE_r=%.2f D3=%.3f ESS=%.2f Q=%.2f\n",
                   q.rmse_ratio, q.d3_self_contradiction, 
                   q.ess_ratio, q.quality_score);
            printf("\n");
            
            oracle_bridge_apply_injection(&bridge, &dec);
            
            /* Update RBPF's Π */
            rbpf_set_pi(&rbpf, 
                        oracle_bridge_get_pi(&bridge),
                        oracle_bridge_get_storvik(&bridge));
            
            injection_count++;
        }
        
        /* Periodic status */
        if (t % 500 == 0 && t > 0) {
            PiQualitySnapshot q = oracle_bridge_get_quality(&bridge);
            printf("[t=%4d] Status: RMSE_r=%.2f D3=%.3f ESS=%.2f Q=%.2f\n",
                   t, q.rmse_ratio, q.d3_self_contradiction,
                   q.ess_ratio, q.quality_score);
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * RESULTS
     *═══════════════════════════════════════════════════════════════════*/
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                        FINAL RESULTS                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    OracleBridgeStats stats = oracle_bridge_get_stats(&bridge);
    
    printf("Injections:           %d\n", stats.total_injections);
    printf("Thompson samples:     %d\n", stats.thompson_samples);
    printf("Storvik self-correct: %d\n", stats.storvik_self_corrections);
    printf("Average gamma:        %.3f\n", stats.avg_gamma);
    printf("\n");
    
    printf("Final Π (Operating):\n");
    const float *Pi_final = oracle_bridge_get_pi(&bridge);
    for (int i = 0; i < K; i++) {
        printf("  [");
        for (int j = 0; j < K; j++) {
            printf(" %.3f", Pi_final[i * K + j]);
        }
        printf(" ]\n");
    }
    printf("\n");
    
    printf("True Π:\n");
    for (int i = 0; i < K; i++) {
        printf("  [");
        for (int j = 0; j < K; j++) {
            printf(" %.3f", market.true_Pi[i * K + j]);
        }
        printf(" ]\n");
    }
    printf("\n");
    
    /* Compute Frobenius error */
    float frob_error = 0;
    for (int i = 0; i < K * K; i++) {
        float d = Pi_final[i] - market.true_Pi[i];
        frob_error += d * d;
    }
    frob_error = sqrt(frob_error);
    printf("Frobenius error (final vs true): %.4f\n", frob_error);
    
    /*═══════════════════════════════════════════════════════════════════
     * CLEANUP
     *═══════════════════════════════════════════════════════════════════*/
    
    oracle_bridge_free(&bridge);
    rbpf_free(&rbpf);
    sim_free(&market);
    
    printf("\nDone.\n");
    
    return 0;
}
