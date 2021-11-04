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


class RMSProp(Optimizer, ABC):
    def __init__(self, params, lr, beta, eps=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": np.zeros_like(layer["dW"])})
            layer.update({"sb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["sW"] = self.beta * param["sW"] + (1 - self.beta) * np.square(param["dW"])
            param["W"] -= self.lr * param["dW"] / np.sqrt(param["sW"] + self.eps)
            param["sb"] = self.beta * param["sb"] + (1 - self.beta) * np.square(param["db"])
            param["b"] -= self.lr * param["db"] / np.sqrt(param["sb"] + self.eps)
