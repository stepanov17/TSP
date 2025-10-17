"""
Example: estimate TSP(a, m, b, p) parameters using an inverse mapping method
"""

import numpy as np
import warnings

from scipy.optimize import minimize

from sample_generator import tsp_sample
from ks_test import ks_test

warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_P: float = 0.1  # minimum value of TSP power parameter
MAX_P: float = 15.  # maximum value of TSP power parameter


def _tsp_invcdf(
        u: np.array, a: float, m: float, b: float, p: float
) -> np.array:
    """
    Quantile function of the TSP distribution.
    """
    q = (m - a) / (b - a)
    return np.where(
        u <= q,
        a + (m - a) * ((b - a) / (m - a) * u) ** (1/p),
        b - (b - m) * ((b - a) / (b - m) * (1 - u)) ** (1/p)
    )


def _fit_tsp_objective(params, sorted_sample, n):
    """
    A functional to minimize
    with transformed parameters to ensure a < m < b
    (inverse mapping method)
    """
    a, m_transformed, b, p = params
    # Transform m_transformed back to ensure a < m < b
    m = a + (b - a) * 1 / (1 + np.exp(-m_transformed))
    u = (np.arange(1, n + 1) - 0.5) / n  # vector of (i - 0.5) / n
    inv_cdf_vals = _tsp_invcdf(u, a, m, b, p)
    return np.sum((inv_cdf_vals - sorted_sample) ** 2)


def tsp_invm_estimate(sample: np.array):
    """
    Estimate TSP(a, m, b, p) parameters
    using an inverse mapping method
    """

    n = len(sample)
    sorted_sample = np.sort(sample)

    # initial guess
    a_init = np.min(sample) - 0.1 * np.std(sample)
    b_init = np.max(sample) + 0.1 * np.std(sample)
    m_init = np.median(sample)
    p_init = 2.0

    # Transform initial m to unconstrained space
    m_transformed_init = np.log((m_init - a_init) / (b_init - m_init))

    bounds = [
        (None, None),  # a
        (None, None),  # m_transformed (unconstrained)
        (None, None),  # b
        (MIN_P, MAX_P)  # p > 0
    ]

    result = minimize(
        fun=_fit_tsp_objective,
        x0=[a_init, m_transformed_init, b_init, p_init],
        args=(sorted_sample, n),
        bounds=bounds,
        method="L-BFGS-B"
    )

    # Extract and transform parameters back
    a_est, m_transformed_est, b_est, p_est = result.x
    m_est = (a_est + (b_est - a_est) *
             1 / (1 + np.exp(-m_transformed_est)))  # sigmoid
    return a_est, m_est, b_est, p_est


if __name__ == "__main__":

    n = 2000  # sample length
    a, m, b, p = 0., 0.7, 1., 2.1   # TSP(a, m, b, p) parameters
    sample = tsp_sample(a, m, b, p, n)

    a_est, m_est, b_est, p_est = tsp_invm_estimate(sample)
    out = [f"{v:.3f}" for v in (a_est, m_est, b_est, p_est)]
    print(f"inv. mapping: {', '.join(out)}")
    print("KS test passed:", ks_test(sample, (a_est, m_est, b_est, p_est)))
