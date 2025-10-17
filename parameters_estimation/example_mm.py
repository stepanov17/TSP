"""
Example: method of moments for TSP(a, m, b, p) parameters
"""

import numpy as np
import warnings

from scipy import stats
from scipy.optimize import minimize
from scipy.special import comb

from sample_generator import tsp_sample
from ks_test import ks_test

warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_P: float = 0.1  # minimum value of TSP power parameter
MAX_P: float = 15.  # maximum value of TSP power parameter


def _tsp_initial_moments(
        a: float, m: float, b: float, p: float, max_k: int = 4
) -> np.array:
    """
    Calculate initial moments nu_k = E x^k for TSP(a, m, b, p)
    """
    moments = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        sum_left, sum_right = 0., 0.
        for i in range(k + 1):
            # a < x <= m
            bin_coeff = comb(k, i)
            term_left = (bin_coeff * (a ** (k - i))
                         * ((m - a) ** (i + 1)) / (i + p))
            sum_left += term_left

            # m <= x < b
            term_right = (bin_coeff * (b ** (k - i)) *
                          ((-1) ** i) *
                          ((b - m) ** (i + 1)) / (i + p))
            sum_right += term_right

        moments[k] = (p / (b - a)) * (sum_left + sum_right)

    return moments


def _tsp_moments(
        a: float, m: float, b: float, p: float) -> np.array:
    """
    Calculate expectation, variance, skewness and kurtosis
    for TSP(a, m, b, p)
    """
    if not (a < m < b):
        raise ValueError(
            f"invalid a, m, b: {a}, {m}, {b}; expected: a < m < b")
    if p <= 0:
        raise ValueError("p must be positive")

    # calculate initial moments nu1, nu2, nu3, nu4
    nu = _tsp_initial_moments(a, m, b, p, max_k=4)
    nu1, nu2, nu3, nu4 = nu[1], nu[2], nu[3], nu[4]

    # central moments
    mu2 = nu2 - nu1 ** 2
    mu3 = nu3 - 3 * nu2 * nu1 + 2 * nu1 ** 3
    mu4 = nu4 - 4 * nu3 * nu1 + 6 * nu2 * nu1 ** 2 - 3 * nu1 ** 4

    skewness = mu3 / (mu2 ** 1.5)
    kurtosis = mu4 / (mu2 ** 2) - 3.

    return np.array([nu1, mu2, skewness, kurtosis])


def _tsp_mom_objective(tsp_params: np.array,
                       sample_moments: list[float],
                       weights: tuple[float] = (1., 1., 20., 20.)):
    """
    Error to minimize while estimating parameters of TSP(a, b, m, p)
    """
    try:
        a, m, b, p = tsp_params
        theory_moments = _tsp_moments(a, m, b, p)
        # normalize errors
        errors = [
            (theory - sample) / max(1, abs(sample))
            for theory, sample in zip(theory_moments, sample_moments)
        ]
        # weighted error
        return sum(w * e ** 2 for w, e in zip(weights, errors))
    except:
        return 1.e12


def tsp_mm_estimate(sample: np.array):
    """
    Estimate parameters of TSP(a, m, b, p) by the method of moments
    """
    sample_moments = [
        np.mean(sample), np.var(sample, ddof=0),
        stats.skew(sample), stats.kurtosis(sample)
    ]

    rng = np.ptp(sample)
    # initial approximation
    a_init = np.min(sample) - 0.05 * rng
    b_init = np.max(sample) + 0.05 * rng
    m_init = np.mean(sample)
    p_init = 2.

    bounds = [
        (None, m_init - 1e-5),           # a
        (a_init + 1e-5, b_init - 1e-5),  # m
        (m_init + 1e-5, None),           # b
        (MIN_P, MAX_P)                   # p
    ]

    result = minimize(
        _tsp_mom_objective,
        x0=np.array([a_init, m_init, b_init, p_init]),
        args=sample_moments,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-8}
    )

    a_est, m_est, b_est, p_est = result.x
    return a_est, m_est, b_est, p_est


if __name__ == "__main__":

    n = 2000  # sample length
    a, m, b, p = 0., 0.7, 1., 2.1   # TSP(a, m, b, p) parameters
    sample = tsp_sample(a, m, b, p, n)

    a_est, m_est, b_est, p_est = tsp_mm_estimate(sample)
    out = [f"{v:.3f}" for v in (a_est, m_est, b_est, p_est)]
    print(f"MM: {', '.join(out)}")
    print("KS test passed:", ks_test(sample, (a_est, m_est, b_est, p_est)))
