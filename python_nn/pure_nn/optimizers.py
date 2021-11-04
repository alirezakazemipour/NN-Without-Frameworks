from abc import ABC
from .utils import mat_add, rescale


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
