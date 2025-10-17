"""
Example: maximum likelihood estimates for TSP(a, m, b, p) parameters
"""

import numpy as np
import warnings

from scipy.optimize import differential_evolution

from sample_generator import tsp_sample
from ks_test import ks_test

warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_P: float = 0.1  # minimum value of TSP power parameter
MAX_P: float = 15.  # maximum value of TSP power parameter


def _log_likelihood(params: tuple, sample: np.array) -> float:
    """
    Log likelihood function for given TSP params (a, m, b, p)
    """
    a, m, b, p = params
    if ((a >= min(sample)) or (b <= max(sample))
            or (m <= a) or (m >= b) or (p <= 0)):
        return -np.inf

    left = sample[sample <= m]
    right = sample[sample >= m]

    eps = 1e-8
    term1 = np.sum(np.log(np.clip((left - a) / (m - a), eps, None)))
    term2 = np.sum(np.log(np.clip((b - right) / (b - m), eps, None)))
    return (len(sample) * np.log(p) -
            len(sample) * np.log(b - a) + (p - 1) * (term1 + term2))


def tsp_mle_estimate(sample: np.array) -> tuple:
    """
    Get MLEs (maximum likelihood estimates)
    for TSP(a, m, b, p) parameters using a given sample
    """
    min_val, max_val = min(sample), max(sample)
    r = max_val - min_val

    # TSP parameters bounds
    bounds = [
        (min_val - 0.5 * r, min_val),  # a
        (min_val, max_val),            # m
        (max_val, max_val + 0.5 * r),  # b
        (MIN_P, MAX_P)                 # p
    ]

    # global optimization
    result = differential_evolution(
        lambda params: -_log_likelihood(params, sample),
        bounds=bounds,
        strategy="best1bin",
        maxiter=5000,
        popsize=15,
        tol=1e-6,
        recombination=0.7,
        seed=42
    )
    a, m, b, p = result.x
    return a, m, b, p


if __name__ == "__main__":

    n = 1000  # sample length
    a, m, b, p = 0., 0.7, 1., 2.1   # TSP(a, m, b, p) parameters
    sample = tsp_sample(a, m, b, p, n)

    a_est, m_est, b_est, p_est = tsp_mle_estimate(sample)
    out = [f"{v:.3f}" for v in (a_est, m_est, b_est, p_est)]
    print(f"MLE: {', '.join(out)}")
    print("KS test passed:", ks_test(sample, (a_est, m_est, b_est, p_est)))
