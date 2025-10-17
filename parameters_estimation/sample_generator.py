import numpy as np


def tsp_sample(a: float, m: float, b: float, p: float, n: int) -> np.array:
    """
    Generate sample of length n from TSP(a, m, b, p) distribution
    """
    u = np.random.uniform(0, 1, n)
    mask = u <= (m - a) / (b - a)
    sample = np.where(
        mask,
        a + (m - a) * ((b - a) / (m - a) * u) ** (1 / p),
        b - (b - m) * ((b - a) / (b - m) * (1 - u)) ** (1 / p)
    )
    return sample
