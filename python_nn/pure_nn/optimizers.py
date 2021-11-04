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
