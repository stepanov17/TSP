from scipy import stats
import numpy as np


def _tsp_cdf(
        x: float, a: float, b: float, m: float, p: float
) -> float:
    """CDF for TSP(a, m, b, p)"""
    if x < a:
        return 0.0
    elif x <= m:
        return ((m - a)/(b - a)) * ((x - a)/(m - a))**p
    elif x < b:
        return 1.0 - ((b - m)/(b - a)) * ((b - x)/(b - m))**p
    else:
        return 1.0


def ks_test(sample: np.array,
            param_estimates: tuple[float],
            alpha: float=0.05) -> bool:
    """
    KS-test to check if a sample fits the TSP distribution
    with the given (estimated) parameters;
    the default significance level (alpha) is 0.05
    """
    s = np.sort(sample)
    n = len(s)
    a_est, m_est, b_est, p_est = param_estimates

    # calculate empirical and theoretical CDF
    ecdf = np.arange(1, n + 1) / n
    tcdf = np.array(
        [_tsp_cdf(x, a_est, b_est, m_est, p_est) for x in s])

    # calculate Kolmogorov-Smirnov statistics
    D_plus = np.max(ecdf - tcdf)
    D_minus = np.max(tcdf - (np.arange(0, n) / n))
    ks_stat = max(D_plus, D_minus)
    p_value = stats.kstwo.sf(ks_stat, n)
    return p_value > alpha
