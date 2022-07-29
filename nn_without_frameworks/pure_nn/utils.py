from typing import Tuple, Union
from numbers import Number
from copy import deepcopy
from more_itertools import sliding_window
from itertools import chain


def broadcast(x, y):
    a, b = deepcopy(x), deepcopy(y)
    if a.rows < b.rows:
        while a.rows < b.rows:
            a.append(a[0])
    else:
        while b.rows < a.rows:
            b.append(b[0])

    if a.cols < b.cols:
        for i in range(a.rows):
            a[i].append(a[i][-1])
    elif b.cols < a.cols:
        for i in range(b.rows):
            b[i].append(b[i][-1])

    assert a.shape == b.shape
    return a, b


def naive_matmul(a, b):
    i, j = a.shape
    n, k = b.shape
    temp = Matrix(i, k)
    for w in range(i):
        for h in range(k):
            temp[w, h] = 0
            for r in range(j):
                temp[w, h] += a[w, r] * b[r, h]
    return temp


def strassen(x, y):
    i, j = x.shape
    n, k = y.shape
    if x.shape == (1, 1):
        return x * y

    a, b = Matrix(x[: i // 2, : j // 2]), Matrix(x[: i // 2, j // 2:])
    c, d = Matrix(x[i // 2:, : j // 2]), Matrix(x[i // 2:, j // 2:])
    e, f = Matrix(y[: n // 2, : k // 2]), Matrix(y[: n // 2, k // 2:])
    g, h = Matrix(y[n // 2:, : k // 2]), Matrix(y[n // 2:, k // 2:])

    m1 = strassen(a + c, e + f)
    m2 = strassen(b + d, g + h)
    m3 = strassen(a - d, e + h)
    m4 = strassen(a, f - h)
    m5 = strassen(c + d, e)
    m6 = strassen(a + b, h)
    m7 = strassen(d, g - e)

    i = m2 + m3 - m6 - m7
    j = m4 + m6
    k = m5 + m7
    l = m1 - m3 - m4 - m5
    return (i.concat(j)).stack(k.concat(l))


class Matrix:
    def __init__(self, *args):
        if len(args) == 2:
            self._rows = args[0]
            self._cols = args[1]
            self._value = [[None for _ in range(self._cols)] for _ in range(self._rows)]

        elif len(args) == 3:
            self._rows = args[0]
            self._cols = args[1]
            self._value = [[args[2] for _ in range(self._cols)] for _ in range(self._rows)]
        else:
            value = args[0] if isinstance(args[0][0], list) else list(args[0])
            self._rows = len(value)
            self._cols = len(value[0])
            self._value = value

        self._shape = self._rows, self._cols

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        cols = self._get_cols()
        return cols

    @property
    def value(self):
        return self._value

    @property
    def shape(self):
        self._update_shape()
        return self._shape

    @value.setter
    def value(self, v):
        self._value = v

    def __getitem__(self, item: Union[int, slice, Tuple]):
        if isinstance(item, (int, slice)):
            return self._value[item]

        elif isinstance(item, Tuple):
            i, j = item
            if isinstance(i, int) and isinstance(j, (int, slice)):
                return self._value[i][j]  # noqa
            elif isinstance(i, slice) and isinstance(j, (int, slice)):
                return [v[j] for v in self._value[i]]
            else:
                raise NotImplementedError(f"{type(item)} -> ({type(i)}, {type(j)})")

        else:
            raise NotImplementedError(type(item))

    def __setitem__(self, key: Union[int, Tuple], value):
        if isinstance(key, int):
            self._value[key] = value
        elif isinstance(key, Tuple):
            i, j = key
            self._value[i][j] = value  # noqa
        else:
            raise NotImplementedError

    def __len__(self):
        return self._rows

    def __matmul__(self, other):
        assert isinstance(other, Matrix)
        i, j = self._rows, self._cols
        n, k = other.rows, other.cols
        assert n == j
        if i == j and n == k:
            temp = strassen(self, other)
        else:
            temp = naive_matmul(self, other)

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
            a, b = broadcast(self, other)
            i, j = a.rows, a.cols
            n, k = b.rows, b.cols

            assert i == n and j == k
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = a.value[w][h] * b[w, h]
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
            a, b = broadcast(self, other)
            i, j = a.rows, a.cols
            n, k = b.rows, b.cols

            assert i == n and j == k
            temp = Matrix(i, j)
            for w in range(i):
                for h in range(j):
                    temp[w, h] = a.value[w][h] + b[w, h]
            return temp

        else:
            raise NotImplementedError(type(other))

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __pow__(self, power, modulo=None):
        i, j = self._rows, self._cols
        temp = Matrix(i, j)
        for w in range(i):
            for h in range(j):
                temp[w, h] = self._value[w][h] ** power
        return temp

    def __repr__(self):
        x = "["
        for i in range(self._rows):
            x += str(self._value[i]) + "\n" if i != self._rows - 1 else str(self._value[i])
        x += "]"
        return x

    def append(self, other):
        self._value.append(other)
        self._update_shape()

    def _update_shape(self):
        self._rows = len(self._value)
        self._cols = len(self._value[0]) if len(self._value) == 1 else self._get_cols()
        self._shape = self._rows, self._cols

    def _get_cols(self):
        cols = len(self._value[0])
        for i in range(self._rows):
            if not cols == len(self._value[i]):
                raise ValueError(f"Number of columns is not consistent in each row! "
                                 f"found {len(self._value[i])} columns in row {i}"
                                 )
        return cols

    def sum(self):
        i, j = self._rows, self._cols
        temp = Matrix([[0 for _ in range(j)]])
        for w in range(i):
            for h in range(j):
                temp[0, h] += self._value[w][h]
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
                temp[0, h] += ((self._value[w][h] - mu[0, h]) ** 2) / i
        return temp

    def concat(self, other):
        assert self._rows == other.rows
        temp = Matrix(self._rows, self._cols + other.cols)
        for i in range(self._rows):
            temp[i] = self._value[i] + other.value[i]
        return temp

    def stack(self, other):
        assert self._cols == other.cols
        return Matrix(self._value + other.value)


if __name__ == "__main__":
    a = Matrix([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
                ]
               )
    b = Matrix([[17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32]
                ]
               )
    print(a, b)
    print(a @ b)
    print(a + b)
    print(a * b)
    print(a.t())
    print(a * (1 / 5))
