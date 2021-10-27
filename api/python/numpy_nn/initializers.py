from abc import ABC

import numpy as np


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class Constant(Initializer, ABC):
    def __init__(self, c=0):
        self._c = c

    def initialize(self, x):
        return self._c * np.ones(x)


class RandomUniform(Initializer, ABC):
    def initialize(self, shape):
        return np.random.uniform(0, 1, shape)