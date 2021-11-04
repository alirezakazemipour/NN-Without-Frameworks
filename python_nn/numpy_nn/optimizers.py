from abc import ABC
import numpy as np


def supported_optimizers():
    return [x.__name__ for x in Optimizer.__subclasses__()]


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def apply(self):
        raise NotImplementedError


class SGD(Optimizer, ABC):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def apply(self):
        for param in self.params.values():
            param["W"] -= self.lr * param["dW"]
            param["b"] -= self.lr * param["db"]


class Momentum(Optimizer, ABC):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"gW": np.zeros_like(layer["dW"])})
            layer.update({"gb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["gW"] = param["dW"] + self.mu * param["gW"]
            param["W"] -= self.lr * param["gW"]
            param["gb"] = param["db"] + self.mu * param["gb"]
            param["b"] -= self.lr * param["gb"]
