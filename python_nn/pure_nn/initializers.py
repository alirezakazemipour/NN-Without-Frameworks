from abc import ABC

import random


def supported_initializers():
    return [x.__name__ for x in Initializer.__subclasses__()]


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class Constant(Initializer, ABC):
    def __init__(self, c=0):
        self._c = c

    def initialize(self, x):
        if isinstance(x[0], int):
            w = len(x)
            temp = [None for _ in range(w)]
            for i in range(w):
                temp[i] = self._c
            return temp
        elif isinstance(x[0], list):
            w, h = len(x), len(x[0])
            temp = [[None for _ in range(h)] for _ in range(w)]
            for i in range(w):
                for j in range(h):
                    temp[i][j] = self._c
            return temp
        else:
            raise TypeError


class RandomUniform(Initializer, ABC):
    def initialize(self, x):
        w, h = len(x), len(x[0])
        temp = [[None for _ in range(h)] for _ in range(w)]
        for i in range(w):
            for j in range(h):
                temp[i][j] = random.uniform(0, 1)
        return temp
