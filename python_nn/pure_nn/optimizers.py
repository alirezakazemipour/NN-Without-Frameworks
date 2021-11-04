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
            grad_step_w = rescale(param["dW"], -self.lr)
            param["W"] = mat_add(param["W"], grad_step_w)
            grad_step_b = rescale(param["db"], -self.lr)
            param["b"] = mat_add(param["b"], grad_step_b)


class Momentum(Optimizer, ABC):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"gW": rescale(layer["dW"], 0)})
            layer.update({"gb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():
            param["gW"] = mat_add(param["dW"], rescale(param["gW"], self.mu))
            grad_step_w = rescale(param["gW"], -self.lr)
            param["W"] = mat_add(param["W"], grad_step_w)
            param["gb"] = mat_add(param["db"], rescale(param["gb"], self.mu))
            grad_step_b = rescale(param["gb"], -self.lr)
            param["b"] = mat_add(param["b"], grad_step_b)


class RMSProp(Optimizer, ABC):
    def __init__(self, params, lr, beta, eps=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": rescale(layer["dW"], 0)})
            layer.update({"sb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():

            grad_square_w = element_wise_mul(param["dW"], param["dW"])
            grad_square_w = rescale(grad_square_w, 1 - self.beta)
            param["sW"] = mat_add(rescale(param["sW"], self.beta), grad_square_w)
            grad_step_w = element_wise_mul(param["dW"], element_wise_rev(mat_sqrt(add_scalar(param["sW"], self.eps))))
            param["W"] = mat_add(param["W"], rescale(grad_step_w, -self.lr))

            grad_square_b = element_wise_mul(param["db"], param["db"])
            grad_square_b = rescale(grad_square_b, 1 - self.beta)
            param["sb"] = mat_add(rescale(param["sb"], self.beta), grad_square_b)
            grad_step_b = element_wise_mul(param["db"], element_wise_rev(mat_sqrt(add_scalar(param["sb"], self.eps))))
            param["b"] = mat_add(param["b"], rescale(grad_step_b, -self.lr))