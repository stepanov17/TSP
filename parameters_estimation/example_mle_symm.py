"""
Example: maximum likelihood estimates for TSP(x0, r, p) parameters
"""

import numpy as np
import warnings

from scipy.optimize import differential_evolution

from sample_generator import tsp_sample
from ks_test import ks_test

warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_P: float = 0.1  # minimum value of TSP power parameter
MAX_P: float = 15.  # maximum value of TSP power parameter


def _log_likelihood_symm(params: tuple, sample: np.array) -> float:
    """
    Log likelihood function for given symmetric TSP params (x0, r, p)
    """
    x0, r, p = params
    a = x0 - r
    b = x0 + r

    if r <= 0 or p <= 0 or a >= min(sample) or b <= max(sample):
        return -np.inf

    left = sample[sample <= x0]
    right = sample[sample >= x0]
    term1 = len(sample) * np.log(p) - len(sample) * np.log(2 * r)
    term2 = (p - 1) * (np.sum(np.log((left - a) / r)) +
                       np.sum(np.log((b - right) / r)))
    return term1 + term2


def tsp_symm_mle_estimate(sample: np.array) -> tuple:
    """
    Get MLEs (maximum likelihood estimates)
    for TSP(x0, r, p) parameters using a given sample
    """
    min_val, max_val = min(sample), max(sample)
    rng = max_val - min_val
    bounds = [
        (min_val, max_val),     # x0
        (0.5 * rng, 10 * rng),  # r
        (MIN_P, MAX_P)          # p
    ]

    result = differential_evolution(
        lambda params: -_log_likelihood_symm(params, sample),
        bounds=bounds,
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        recombination=0.7,
        seed=42
    )
    x0, r, p = result.x
    return x0, r, p


if __name__ == "__main__":

    n = 1000  # sample length
    x0, r, p = 0.3, 0.7, 2.1   # TSP(x0, r, p) parameters
    sample = tsp_sample(x0 - r, x0, x0 + r, p, n)

    x0_est, r_est, p_est = tsp_symm_mle_estimate(sample)
    out = [f"{v:.3f}" for v in (x0_est, r_est, p_est)]
    print(f"MLE symm: {', '.join(out)}")
    print("KS test passed:", ks_test(sample, (x0_est - r_est, x0_est, x0_est + r_est, p_est)))
