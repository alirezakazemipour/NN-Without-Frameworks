import numpy as np


def binary_cross_entropy(p, t):
    return t * np.log(p) + (1 - t) * np.log(1 - p)
