from typing import Tuple, Union
from numbers import Number
from copy import deepcopy


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
            value = args[0]
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

    def __getitem__(self, item: Union[int, Tuple]):
        if isinstance(item, Tuple):
            i, j = item
            return self._value[i][j]  # noqa
        elif isinstance(item, int):
            return self._value[item]
        else:
            raise NotImplementedError

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


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    print(a, b)
    # print(a @ b)
    print(a + b)
    print(a * b)
    # print(a.t())
    # print(a * (1 / 5))
