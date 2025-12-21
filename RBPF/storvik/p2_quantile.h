/**
 * @file p2_quantile.h
 * @brief P² Algorithm for Online Quantile Estimation
 *
 * Implements Jain & Chlamtac (1985) "The P² Algorithm for Dynamic
 * Calculation of Quantiles and Histograms Without Storing Observations"
 *
 * Properties:
 *   - O(1) time per observation
 *   - O(1) space (5 markers regardless of data size)
 *   - Distribution-free (no Gaussian assumption)
 *   - Self-calibrating (adapts to observed data)
 *
 * Usage:
 *   P2Quantile q;
 *   p2_init(&q, 0.999);  // Track 99.9th percentile
 *   
 *   for each observation x:
 *       p2_update(&q, x);
 *       double p999 = p2_get_quantile(&q);
 *
 * Reference: Communications of the ACM, Vol 28, No 10, October 1985
 */

#ifndef P2_QUANTILE_H
#define P2_QUANTILE_H

#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * P² QUANTILE ESTIMATOR
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double q[5];      /* Marker heights (quantile estimates) */
    int n[5];         /* Marker positions (1-indexed observation counts) */
    double np[5];     /* Desired marker positions (floating point) */
    double dn[5];     /* Increment for desired positions */
    double p;         /* Target percentile (e.g., 0.999) */
    int count;        /* Total observations seen */
    int initialized;  /* 1 after first 5 observations */
} P2Quantile;

/**
 * @brief Initialize P² estimator for a target percentile
 *
 * @param pq  P² quantile structure
 * @param p   Target percentile (0.0 - 1.0), e.g., 0.999 for 99.9th percentile
 */
static inline void p2_init(P2Quantile *pq, double p)
{
    memset(pq, 0, sizeof(*pq));
    pq->p = p;
    
    /* Desired position increments for markers 0,1,2,3,4 */
    pq->dn[0] = 0.0;
    pq->dn[1] = p / 2.0;
    pq->dn[2] = p;
    pq->dn[3] = (1.0 + p) / 2.0;
    pq->dn[4] = 1.0;
}

/**
 * @brief Parabolic interpolation formula (P-P² formula)
 */
static inline double p2_parabolic(
    double qm1, double q0, double qp1,
    int nm1, int n0, int np1,
    int d)
{
    double a = (double)d / (double)(np1 - nm1);
    double b = (double)(n0 - nm1 + d) * (qp1 - q0) / (double)(np1 - n0);
    double c = (double)(np1 - n0 - d) * (q0 - qm1) / (double)(n0 - nm1);
    return q0 + a * (b + c);
}

/**
 * @brief Linear interpolation fallback
 */
static inline double p2_linear(double q0, double q1, int n0, int n1, int d)
{
    return q0 + (double)d * (q1 - q0) / (double)(n1 - n0);
}

/**
 * @brief Update estimator with new observation
 *
 * @param pq  P² quantile structure
 * @param x   New observation
 */
static inline void p2_update(P2Quantile *pq, double x)
{
    pq->count++;
    
    /* Initial phase: collect first 5 observations */
    if (pq->count <= 5)
    {
        pq->q[pq->count - 1] = x;
        
        if (pq->count == 5)
        {
            /* Sort the 5 initial observations (insertion sort) */
            for (int i = 1; i < 5; i++)
            {
                double key = pq->q[i];
                int j = i - 1;
                while (j >= 0 && pq->q[j] > key)
                {
                    pq->q[j + 1] = pq->q[j];
                    j--;
                }
                pq->q[j + 1] = key;
            }
            
            /* Initialize marker positions */
            for (int i = 0; i < 5; i++)
            {
                pq->n[i] = i + 1;  /* 1, 2, 3, 4, 5 */
            }
            
            /* Initialize desired positions */
            pq->np[0] = 1.0;
            pq->np[1] = 1.0 + 2.0 * pq->p;
            pq->np[2] = 1.0 + 4.0 * pq->p;
            pq->np[3] = 3.0 + 2.0 * pq->p;
            pq->np[4] = 5.0;
            
            pq->initialized = 1;
        }
        return;
    }
    
    /* Main algorithm: update markers */
    
    /* Step 1: Find cell k such that q[k-1] <= x < q[k] */
    int k;
    if (x < pq->q[0])
    {
        pq->q[0] = x;
        k = 1;
    }
    else if (x < pq->q[1])
    {
        k = 1;
    }
    else if (x < pq->q[2])
    {
        k = 2;
    }
    else if (x < pq->q[3])
    {
        k = 3;
    }
    else if (x <= pq->q[4])
    {
        k = 4;
    }
    else
    {
        pq->q[4] = x;
        k = 4;
    }
    
    /* Step 2: Increment positions of markers k+1 through 4 */
    for (int i = k; i < 5; i++)
    {
        pq->n[i]++;
    }
    
    /* Update desired positions */
    for (int i = 0; i < 5; i++)
    {
        pq->np[i] += pq->dn[i];
    }
    
    /* Step 3: Adjust marker heights if necessary */
    for (int i = 1; i <= 3; i++)
    {
        double d = pq->np[i] - (double)pq->n[i];
        
        if ((d >= 1.0 && pq->n[i + 1] - pq->n[i] > 1) ||
            (d <= -1.0 && pq->n[i - 1] - pq->n[i] < -1))
        {
            int di = (d >= 0) ? 1 : -1;
            
            /* Try parabolic interpolation */
            double qnew = p2_parabolic(
                pq->q[i - 1], pq->q[i], pq->q[i + 1],
                pq->n[i - 1], pq->n[i], pq->n[i + 1],
                di);
            
            /* Check if parabolic result is within bounds */
            if (pq->q[i - 1] < qnew && qnew < pq->q[i + 1])
            {
                pq->q[i] = qnew;
            }
            else
            {
                /* Use linear interpolation */
                int ni = (di == 1) ? i + 1 : i - 1;
                pq->q[i] = p2_linear(pq->q[i], pq->q[ni], pq->n[i], pq->n[ni], di);
            }
            
            pq->n[i] += di;
        }
    }
}

/**
 * @brief Get current quantile estimate
 *
 * @param pq  P² quantile structure
 * @return    Estimated quantile value, or NAN if < 5 observations
 */
static inline double p2_get_quantile(const P2Quantile *pq)
{
    if (!pq->initialized)
    {
        return (pq->count > 0) ? pq->q[pq->count - 1] : 0.0 / 0.0; /* NAN */
    }
    return pq->q[2];  /* Middle marker is the quantile estimate */
}

/**
 * @brief Get observation count
 */
static inline int p2_get_count(const P2Quantile *pq)
{
    return pq->count;
}

/**
 * @brief Check if estimator is warmed up (>= 5 observations)
 */
static inline int p2_is_initialized(const P2Quantile *pq)
{
    return pq->initialized;
}

/**
 * @brief Reset estimator to initial state
 */
static inline void p2_reset(P2Quantile *pq)
{
    double p = pq->p;
    p2_init(pq, p);
}

/**
 * @brief Get all 5 markers for diagnostics
 */
static inline void p2_get_markers(const P2Quantile *pq, 
                                   double *q_out, int *n_out)
{
    for (int i = 0; i < 5; i++)
    {
        if (q_out) q_out[i] = pq->q[i];
        if (n_out) n_out[i] = pq->n[i];
    }
}

#ifdef __cplusplus
}
#endif

#endif /* P2_QUANTILE_H */
