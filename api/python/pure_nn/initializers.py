from abc import ABC

import random


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class Constant(Initializer, ABC):
    def __init__(self, c=0):
        self._c = c

    def initialize(self, x):
        assert len(x) > 0
        w = len(x)
        temp = [None for _ in range(w)]
        for i in range(w):
            temp[i] = self._c
        return temp


class RandomUniform(Initializer, ABC):
    def initialize(self, x):
        assert len(x) > 0
        w, h = len(x), len(x[0])
        temp = [[None for _ in range(h)] for _ in range(w)]
        for i in range(w):
            for j in range(h):
                temp[i][j] = random.uniform(0, 1)
        return temp
