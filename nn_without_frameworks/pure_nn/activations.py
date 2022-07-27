from abc import ABC
import math
from .utils import Matrix


def supported_activations():
    return [x.__name__ for x in Activation.__subclasses__()]


class Activation:
    def __call__(self, x: Matrix):
        return self.forward(x)

    def forward(self, x: Matrix) -> Matrix:
        raise NotImplementedError

    def derivative(self, x: Matrix) -> Matrix:
        raise NotImplementedError


class Linear(Activation, ABC):
    def forward(self, x: Matrix) -> Matrix:
        return x

    def derivative(self, x: Matrix) -> Matrix:
        w, h = x.rows, x.cols
        temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = 1
        return temp


class ReLU(Activation, ABC):
    def forward(self, x: Matrix) -> Matrix:
        w, h = x.rows, x.cols
        temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = x[i, j] if x[i, j] > 0 else 0
        return temp

    def derivative(self, x: Matrix) -> Matrix:
        assert len(x) > 0
        w, h = x.rows, x.cols
        temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = 1 if x[i, j] > 0 else 0
        return temp


class Tanh(Activation, ABC):
    def forward(self, x: Matrix) -> Matrix:
        assert len(x) > 0
        w, h = x.rows, x.cols
        temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = math.tanh(x[i, j])
        return temp

    def derivative(self, x: Matrix) -> Matrix:
        temp = self.forward(x)
        x = [[] for _ in range(len(temp))]
        for i, k in enumerate(temp):
            for j in k:
                x[i].append(1 - j ** 2)
        return Matrix(x)
