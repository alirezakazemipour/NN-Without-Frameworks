from abc import ABC
import numpy as np


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
