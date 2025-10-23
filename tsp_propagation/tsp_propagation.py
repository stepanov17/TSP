import datetime
import logging
import math
import json
import os
import sys

from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)-9s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def tsp_inv_cdf(y: float,
                a: float,
                m: float,
                b: float,
                p: float,
                q: float) -> float:
    """inverse CDF(y) for TSP(a, m, b, p) distribution;
    q = (m - a) / (b - a)"""

    if y <= q:
        return a + math.pow(
            y * math.pow(m - a, p - 1.) * (b - a), 1. / p)
    else:
        return b - math.pow(
            (1. - y) * math.pow(b - m, p - 1.) * (b - a), 1. / p)


def get_tsp_sample(
        a: float, m: float, b: float, p: float, n: int) -> list[float]:
    """generate a sample from TSP(a, m, b, p) of size n"""

    q = (m - a) / (b - a)
    u = np.random.uniform(size=n)
    return [tsp_inv_cdf(y, a, m, b, p, q) for y in u]


def substitute_subexpressions(subexpressions: list[str],
                              transform_expression: str) -> str:
    """substitute subexpressions into a transform formula"""

    if not subexpressions:
        return transform_expression

    for se in subexpressions:
        # avoid circular dependencies
        if "subexpressions" in "".join(se.split()):
            raise ValueError(
                "subexpression should not include subexpressions")

    res = transform_expression.strip().replace("\t", " ")

    to_replace = ["subexpressions ", "[ ", " ]"]
    # e.g., "subexpressions [ 1 ]" -> "subexpressions[1]",
    # for the further correct parsing
    for s in to_replace:
        s_new = s.strip()
        while s in res:
            res = res.replace(s, s_new)

    for i in range(len(subexpressions)):
        res = res.replace(f"subexpressions[{i}]", f"({subexpressions[i]})")

    return res


def check_inputs(inputs: dict[str, dict[str, float]]) -> None:
    """check input parameters"""

    tsp_param_names = ["a", "b", "m", "p"]
    normal_param_names = ["m", "sigma"]

    for name, params in inputs.items():
        if not name[:1].isalpha():
            raise ValueError(f"{name}: name must start with a letter")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"invalid input name: {name};"
                             "the name can only consist of numbers, "
                             "letters and underscore")

        distr_type = params.get("type")
        if distr_type is None:
            log.warning(
                f"no distribution type for {name} is set; assuming TSP")
            distr_type = "tsp"

        if distr_type not in ("tsp", "norm"):
            raise ValueError(f"invalid type: {distr_type}, "
                             f"expected to be 'tsp' or 'norm'")

        param_names = sorted(params.keys())
        param_names.remove("type")

        if distr_type == "tsp":

            if param_names != tsp_param_names:
                raise ValueError(f"unexpected param names: {param_names}; "
                                 f"expected: {tsp_param_names}")
            if params["p"] <= 0:
                raise ValueError(f"non-positive p value for {name}")
            if not params["a"] <= params["m"] <= params["b"]:
                raise ValueError(
                    f"condition a <= m <= b is not met for {name}")
            if params["a"] >= params["b"]:
                raise ValueError(f"condition a < b is not met for {name}")

        else:  # normal distribution

            if param_names != normal_param_names:
                raise ValueError(f"unexpected param names: {param_names}; "
                                 f"expected: {tsp_param_names}")
            if params["sigma"] <= 0:
                raise ValueError(
                    f"non-positive sigma value for {name}: {params['sigma']}")


def get_shortest_coverage_interval(
        sorted_data: list[float], p0: float) -> list:
    """get the shortest coverage interval for the probability level of p0"""

    n1 = round(len(sorted_data) * p0)
    n2 = len(sorted_data) - n1

    l0 = float("inf")
    a, b = None, None

    for start in range(n2 + 1):

        c1, c2 = sorted_data[start], sorted_data[start + n1 - 1]
        l = c2 - c1
        if l < l0:
            a, b = c1, c2

    return [a, b]


def add_output_stats(data: list[float], output: dict) -> None:
    """append stats for the transformed distribution"""

    n_digs = output["n_digs"]

    results = {
        "min": round(min(data), n_digs),
        "max": round(max(data), n_digs),
        "mean": round(mean(data), n_digs),
        "stdev": round(stdev(data), n_digs)
    }

    title = "y: " + ", ".join(
        [f"{key} = {value}" for key, value in results.items()])
    plt.title(title)

    # calculate coverage interval
    data.sort()
    n = len(data)

    cov_interval_types = (
        "probabilistically_symmetric", "left", "right", "shortest")
    coverage_interval_type = results.get(
        "coverage_interval", "probabilistically_symmetric")

    coverage_probability = results.get("coverage_probability", 0.95)
    if not 0.5 <= coverage_probability < 1:
        raise ValueError(
            f"invalid coverage probability value: {coverage_probability}; "
            "expecting to be in the range [0.5, 1)")

    if coverage_interval_type == "probabilistically_symmetric":
        dp = 0.5 * (1. - coverage_probability)
        coverage_interval = [data[round(n * dp)], data[round(n * (1 - dp))]]
    elif coverage_interval_type == "left":
        coverage_interval = [data[0], data[round(n * coverage_probability)]]
    elif coverage_interval_type == "right":
        coverage_interval = [
            data[round(n * (1 - coverage_probability))], data[-1]]
    elif coverage_interval_type == "shortest":
        coverage_interval = get_shortest_coverage_interval(
            data, coverage_probability)
    else:
        raise ValueError(
            f"invalid coverage interval type: {coverage_interval_type}; "
            f"expected to be one of {cov_interval_types}")

    results["coverage_interval"] = [round(coverage_interval[0], n_digs),
                                    round(coverage_interval[1], n_digs)]

    output["results"] = results


def transform(annotation: dict) -> list[float]:
    """transform the input distributions"""

    user_var_names = []

    n_trials = int(annotation.get("n_trials", 1000000))  # may be float: 2e6
    min_n_trials = 10000
    if n_trials < min_n_trials:
        log.warning(f"n(trials) is too low, setting to {min_n_trials}")
        n_trials = min_n_trials
    local_vars = locals()

    inputs = annotation["inputs"]
    check_inputs(inputs)

    for input_name, input_distr in inputs.items():
        # check if already in locals
        if input_name in local_vars:
            raise ValueError(
                f"error: name '{input_name}' is already reserved")
        # TODO: checks
        user_var_names.append(input_name)
        if input_distr["type"] == "tsp":
            local_vars[input_name] = get_tsp_sample(
                input_distr["a"], input_distr["m"],
                input_distr["b"], input_distr["p"], n_trials)
        else:  # normal
            local_vars[input_name] = np.random.normal(
                loc=input_distr["m"],
                scale=input_distr["sigma"],
                size=n_trials)

    user_vars_tuple = f"({', '.join(user_var_names)})"
    user_vars_for = f"for {user_vars_tuple} in zip{user_vars_tuple}"

    transform_expr = substitute_subexpressions(
        annotation.get("subexpressions", []), annotation["transform"])
    distr_transform = f"[{transform_expr} {user_vars_for}]"
    res = eval(distr_transform)
    return res


def add_histogram(data: list, n_digs: int, n_bins: int, output: dict) -> None:
    """histogram: plot and add to output json if required"""

    hist_data = {}

    freqs, bins, _ = plt.hist(data, bins=n_bins, color="lightsteelblue")

    # draw coverage interval and mean
    coverage_interval = output.get("results", {}).get("coverage_interval", [])
    if coverage_interval:
        plt.plot(coverage_interval, [0., 0.], linewidth=7., color="red")
    out_mean = output.get("results", {}).get("mean")
    if out_mean is not None:
        max_y = plt.gca().get_ylim()[1]
        plt.plot([out_mean, out_mean], [0., 0.75 * max_y],
                 linewidth=2., color="red", linestyle="dotted")
    plt.grid()

    for (val, freq) in zip(bins, freqs):
        hist_data[round(val, n_digs)] = int(freq)

    bin_fmt = f"%.{n_digs}f"
    if output.get("add_histogram", True):
        output.setdefault("results", {})["histogram"] = (
            dict(zip([bin_fmt % round(b, n_digs) for b in bins],
                     [int(f) for f in freqs])))


def main(input_file: str):

    try:
        with open(input_file, "r") as f:
            annotation = json.load(f)

        log.info("start calculations")

        transformed_data = transform(annotation)

        out = annotation.copy()
        add_output_stats(transformed_data, out)
        add_histogram(transformed_data, annotation.get("n_digs", 3),
                      annotation.get("n_histogram_bins", 100), out)

        file_name, file_extension = os.path.splitext(input_file)
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out_json = f"{file_name}_results_{suffix}{file_extension}"
        out_png = f"{file_name}_results_{suffix}.png"
        log.info(f"done, the output will be saved as: {out_json}, {out_png}")
        with open(out_json, "w") as f:
            json.dump(out, f, indent=4)
        plt.gcf().set_size_inches(10, 7)
        plt.savefig(out_png, dpi=200)

    except Exception as e:
        log.error(f"error: {e}")


if __name__ == "__main__":

    main("./example.json")


if __name__ == "__0main__":

    args = sys.argv
    if len(args) < 2:
        log.error("no input JSON file provided")
        exit(1)
    main(args[1])
