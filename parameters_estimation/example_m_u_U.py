"""
Example: estimate TSP(a, m, b, p) parameters using mean, uncertainty and coverage interval
"""
import math

import numpy as np
from scipy.optimize import fsolve


def param_equations(variables: list[float],
                    x0: float,
                    u: float,
                    c1: float,
                    c2: float,
                    p0: float) -> list:
    """
    Prepare a list of equations for "coverage interval boundaries" case
    (get (a, m, b, p) by (x0, u, c1, c2, p0))
    """

    a, m, b, p = variables
    # equation (1): x0
    eq1 = (a + (p - 1) * m + b) / (p + 1) - x0
    # equation (2): u
    term = (m - a) * (b - m) / ((b - a)**2)
    numerator = p - 2. * (p - 1) * term
    denominator = (p + 1)**2 * (p + 2)
    eq2 = (b - a)**2 * numerator / denominator - u**2
    # equation (3): c1
    eq3 = a + (m - a) * math.pow(
        0.5 * (b - a) / (m - a) * (1 - p0), 1. / p) - c1
    # equation (4): c2
    eq4 = b - (b - m) * math.pow(
        0.5 * (b - a) / (b - m) * (1 - p0), 1. / p) - c2
    return [eq1, eq2, eq3, eq4]


def estimate_tsp_params(x0: float,
                        u: float,
                        c1: float,
                        c2: float,
                        p0: float) -> tuple:
    """
    Try to get a, b, m, p parameters of TSP(a, b, m, p)
    basing on its mean x0, uncertainty (standard deviation) u
    and coverage interval (c1, c2)
    (corresponding to the probability level of p0);
    """
    initial_approximation = np.array(
        [c1, 0.5 * (c1 + c2), c2, 2.], dtype=np.float64)
    res = fsolve(
        param_equations, initial_approximation, args=(x0, u, c1, c2, p0), xtol=1.e-5)
    a_est, m_est, b_est, p_est = res.tolist()

    if not all([p_est > 0, a_est < m_est < b_est]):
        raise ArithmeticError("cannot fit TSP parameters")

    if (b_est - a_est) / (c2 - c1) > 1e2:  # too wide range
        raise ArithmeticError("cannot fit TSP parameters")

    return a_est, m_est, b_est, p_est


if __name__ == "__main__":

    a_est, m_est, b_est, p_est = estimate_tsp_params(0., 1.1, -2.2, 2., 0.95)
    out = [f"{v:.3f}" for v in (a_est, m_est, b_est, p_est)]
    print(f"TSP parameter estimates: {', '.join(out)}")
