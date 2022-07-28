from abc import ABC
from .utils import *


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
            grad_step_w = param["dW"] * -self.lr
            param["W"] += grad_step_w
            grad_step_b = param["db"] * -self.lr
            param["b"] += grad_step_b


class Momentum(Optimizer, ABC):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"gW": layer["dW"] * 0})
            layer.update({"gb": layer["db"] * 0})

    def apply(self):
        for param in self.params.values():
            param["gW"] = param["dW"] + (param["gW"] * self.mu)
            grad_step_w = param["gW"] * -self.lr
            param["W"] += grad_step_w
            param["gb"] = param["db"] + (param["gb"] * self.mu)
            grad_step_b = param["gb"] * -self.lr
            param["b"] += grad_step_b


class RMSProp(Optimizer, ABC):
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": layer["dW"] * 0})
            layer.update({"sb": layer["db"] * 0})

    def apply(self):
        for param in self.params.values():
            grad_square_w = param["dW"] * param["dW"]
            grad_square_w = grad_square_w * (1 - self.beta)
            param["sW"] = (param["sW"] * self.beta) + grad_square_w
            grad_step_w = param["dW"] * ((param["sW"] ** 0.5) + self.eps) ** -1
            param["W"] += (grad_step_w * -self.lr)

            grad_square_b = param["db"] * param["db"]
            grad_square_b = grad_square_b * (1 - self.beta)
            param["sb"] = (param["sb"] * self.beta) + grad_square_b
            grad_step_b = param["db"] * (((param["sb"] ** 0.5) + self.eps) ** -1)
            param["b"] += (grad_step_b * -self.lr)


class AdaGrad(Optimizer, ABC):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super(AdaGrad, self).__init__(params, lr)
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": layer["dW"] * 0})
            layer.update({"sb": layer["db"] * 0})

    def apply(self):
        for param in self.params.values():
            grad_square_w = param["dW"] ** 2
            param["sW"] += grad_square_w
            grad_step_w = param["dW"] * (((param["sW"] ** 0.5) + self.eps) ** -1)
            param["W"] += (grad_step_w * -self.lr)

            grad_square_b = param["db"] ** 2
            param["sb"] += grad_square_b
            grad_step_b = param["db"] * ((param["sb"] ** 0.5) + self.eps) ** -1
            param["b"] += (grad_step_b * -self.lr)


class Adam(Optimizer, ABC):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"mW": layer["dW"] * 0})
            layer.update({"vW": layer["dW"] * 0})
            layer.update({"mb": layer["db"] * 0})
            layer.update({"vb": layer["db"] * 0})

    def apply(self):
        for param in self.params.values():
            param["mW"] = (param["dW"] * (1 - self.beta1)) + (param["mW"] * self.beta1)
            param["vW"] = (param["dW"] ** 2) * (1 - self.beta2) + (param["vW"] * self.beta2)
            mw_hat = param["mW"] * (1 / (1 - self.beta1 ** self.k))
            vw_hat = param["vW"] * (1 / (1 - self.beta2 ** self.k))
            grad_step_w = mw_hat * ((vw_hat ** 0.5) + self.eps) ** -1
            param["W"] += (grad_step_w * -self.lr)

            param["mb"] = (param["db"] * (1 - self.beta1)) + (param["mb"] * self.beta1)
            param["vb"] = ((param["db"] ** 2) * (1 - self.beta2)) + (param["vb"] * self.beta2)
            mb_hat = param["mb"] * (1 / (1 - self.beta1 ** self.k))
            vb_hat = param["vb"] * (1 / (1 - self.beta2 ** self.k))
            grad_step_b = mb_hat * ((vb_hat ** 0.5) + self.eps) ** -1
            param["b"] += (grad_step_b * -self.lr)
        self.k += 1
