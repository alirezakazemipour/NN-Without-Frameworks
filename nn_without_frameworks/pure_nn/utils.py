from copy import deepcopy
from typing import Tuple
from numbers import Number


def equal_batch_size(a, b):
    w0, w1 = len(a), len(b)
    if w0 < w1:
        w = w1
        temp = deepcopy(a)
        x = a
    else:
        w = w0
        temp = deepcopy(b)
        x = b
    while len(temp) < w:
        temp.append(x[0])
    if w0 < w1:
        return temp, b
    else:
        return a, temp


class Matrix:
    def __init__(self, *args):
        if len(args) > 1:
            self._rows = args[0]
            self._cols = args[1]
            self._value = [[None for _ in range(self._cols)] for _ in range(self._rows)]

        else:
            value = args[0]
            self._rows = len(value)
            self._cols = len(value[0])
            self._value = value

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def __getitem__(self, item: Tuple):
        i, j = item
        return self._value[i][j]  # noqa

    def __setitem__(self, key: Tuple, value):
        i, j = key
        self._value[i][j] = value  # noqa

    def __len__(self):
        return self._rows

    def __matmul__(self, other):
        assert isinstance(other, Matrix)
        i, j = self._rows, self._cols
        n, k = other.rows, other.cols

        assert n == j
        temp = Matrix(i, k)
        for w in range(i):
            for h in range(k):
                temp[w, h] = 0
                for r in range(j):
                    temp[w, h] += self._value[w][r] * other[r, h]
        return temp

    def __mul__(self, other):
        if isinstance(other, Number):
            i, j = self._rows, self._cols
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = self._value[w][h] * other
            return temp

        elif isinstance(other, Matrix):
            i, j = self._rows, self._cols
            n, k = other.rows, other.cols

            assert i == n and j == k
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = self._value[w][h] * other[w, h]
            return temp

        else:
            raise NotImplementedError(type(other))

    def __add__(self, other):
        if isinstance(other, Number):
            i, j = self._rows, self._cols
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = self._value[w][h] + other
            return temp

        elif isinstance(other, Matrix):
            i, j = self._rows, self._cols
            n, k = other.rows, other.cols

            assert i == n and j == k
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = self._value[w][h] + other[w, h]
            return temp

        else:
            raise NotImplementedError(type(other))

    def __pow__(self, power, modulo=None):
        i, j = self._rows, self._cols
        temp = Matrix(i, j)
        for w in range(i):
            for h in range(j):
                temp[w, h] = self._value[i][j] ** power
        return temp

    def __repr__(self):
        return str(self.value)

    def sum(self):
        i, j = self._rows, self._cols
        temp = Matrix([[0 for _ in range(j)]])
        for w in range(i):
            for h in range(j):
                temp[0, h] += a[w][h]
        return temp

    def t(self):
        i, j = self._rows, self._cols
        temp = Matrix(j, i)
        for w in range(j):
            for h in range(i):
                temp[w, h] = self._value[h][w]
        return temp

    def mean(self):
        i, j = self._rows, self._cols
        temp = Matrix([[0 for _ in range(j)]])
        for w in range(i):
            for h in range(j):
                temp[0, h] += (self._value[w][h] / i)
        return temp

    def var(self):
        i, j = self._rows, self._cols
        temp = Matrix([[0 for _ in range(j)]])
        mu = self.mean()
        for w in range(i):
            for h in range(j):
                temp[0, h] += ((a[w][h] - mu[0, h]) ** 2) / i
        return temp


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    print(a @ b)
    print(a + b)
    print(a * b)
    print(a.t())
    print(a * (1 / 5))
