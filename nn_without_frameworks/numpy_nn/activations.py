from abc import ABC
import numpy as np


def supported_activations():
    return [x.__name__ for x in Activation.__subclasses__()]


class Activation:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Linear(Activation, ABC):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation, ABC):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class Tanh(Activation, ABC):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.forward(x) ** 2


class Sigmoid(Activation, ABC):
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    def forward(self, x):
        """Numerically stable sigmoid function."""
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))
