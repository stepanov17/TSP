import argparse
import math
import sys

import numpy as np
from scipy.optimize import fsolve


def equations_fixed_boundaries(variables: list[float],
                               x0: float,
                               u: float,
                               a: float,
                               b: float) -> list:
    """
    Prepare a list of equations for "fixed boundaries"
    case (get (m, p) by (x0, u))
    """

    m, p = variables
    # equation (1): x0
    eq1 = (a + (p - 1) * m + b) / (p + 1) - x0
    # equation (2): u
    term = (m - a) * (b - m) / ((b - a)**2)
    numerator = p - 2. * (p - 1) * term
    denominator = (p + 1)**2 * (p + 2)
    eq2 = (b - a)**2 * numerator / denominator - u**2
    return [eq1, eq2]


def equations_coverage_interval(variables: list[float],
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


def get_tsp_params_by_mean_u(
        x0: float, u: float, a: float, b: float, n_digs: int) -> None:
    """
    Try to get a, b, m, p parameters of TSP(a, b, m, p)
    basing on its mean x0, uncertainty (standard deviation) u
    and coverage interval (c1, c2)
    (corresponding to the probability level of p0);
    n_digs is a user-defined number of approximation decimal digits
    """
    initial_approximation = np.array([0.5 * (a + b), 2.], dtype=np.float64)
    values = fsolve(equations_fixed_boundaries,
                    initial_approximation,
                    args=(x0, u, a, b),
                    xtol=math.pow(10., -n_digs-1))

    res = dict(zip(["m", "p"], values))

    if not all([res["p"] > 0, a <= res["m"] <= b]):
        raise ArithmeticError("cannot fit TSP parameters")

    out = []
    for (name, value) in res.items():
        out.append(f"{name} = {value:.{n_digs}f}")
    print(f"TSP parameters: {', '.join(out)}")


def get_tsp_params_by_cov_interval_mean_u(x0: float,
                                          u: float,
                                          c1: float,
                                          c2: float,
                                          p0: float,
                                          n_digs: int) -> None:
    """
    Try to get a, b, m, p parameters of TSP(a, b, m, p)
    basing on its mean x0, uncertainty (standard deviation) u
    and coverage interval (c1, c2)
    (corresponding to the probability level of p0);
    n_digs is a user-defined number of approximation decimal digits
    """
    initial_approximation = np.array(
        [c1, 0.5 * (c1 + c2), c2, 2.], dtype=np.float64)
    values = fsolve(equations_coverage_interval,
                    initial_approximation,
                    args=(x0, u, c1, c2, p0),
                    xtol=math.pow(10., -n_digs-1))

    res = dict(zip(["a", "m", "b", "p"], values))

    if not all([res["p"] > 0, res["a"] <= res["m"] <= res["b"]]):
        raise ArithmeticError("cannot fit TSP parameters")

    if (res["b"] - res["a"]) / (c2 - c1) > 1e2:  # too wide range
        raise ArithmeticError("cannot fit TSP parameters")

    out = []
    for (name, value) in res.items():
        out.append(f"{name} = {value:.{n_digs}f}")
    print(f"TSP parameters: {', '.join(out)}")


def get_args() -> dict:
    """
    Parse and check input args
    :return: the parameters dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries",
                        type=float, nargs=2, required=True,
                        help="boundaries: fixed if a coverage "
                             "probability p0 is not set; otherwise "
                             "the boundarias of a coverage interval")
    parser.add_argument("--x0", type=float,
                        required=True, help="mean value")
    parser.add_argument("--u", type=float,
                        required=True, help="uncertainty value (stdev)")
    parser.add_argument("--p0", type=float,
                        default=None, help="coverage probability")
    parser.add_argument("--ndigs", type=int, default=3,
                        help="number of approximation decimal digits")
    args = parser.parse_args()

    x0 = args.x0
    u = args.u
    if not u > 0.:
        raise ValueError("u must be positive")

    boundaries = args.boundaries
    boundaries.sort()
    b1, b2 = boundaries
    if math.isclose(b1, b2):
        raise ValueError("the boundaries must not coincide")

    if not (b1 <= x0 <= b2):
        raise ValueError("x0 is expected to lie inside the boundaries")

    p0 = args.p0
    if p0 is not None and not (0.5 <= p0 < 1.):
        raise ValueError("p0 is expected to lie inside the interval [0.5, 1)")

    res = {"x0": x0, "u": u}
    if p0 is None:
        res.update({"a": b1, "b": b2})
    else:
        res.update({"c1": b1, "c2": b2, "p0": p0})

    # adjust ndigs
    n_digs = args.ndigs
    r = min(b2 - b1, u)
    n_digs = max(n_digs, 1 + math.ceil(-math.log10(r)))
    res["n_digs"] = n_digs

    return res


def main(args: dict) -> None:
    """
    Do the calculations
    """
    x0, u, n_digs = args["x0"], args["u"], args["n_digs"]
    if "p0" in args:
        get_tsp_params_by_cov_interval_mean_u(
            x0=x0, u=u, c1=args["c1"], c2=args["c2"],
            p0=args["p0"], n_digs=n_digs)
    else:
        get_tsp_params_by_mean_u(
            x0=x0, u=u, a=args["a"], b=args["b"], n_digs=n_digs)


if __name__ == "__main__":

    try:
        main(get_args())
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        exit(1)
