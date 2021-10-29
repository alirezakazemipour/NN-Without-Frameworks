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
            grad_step_w = rescale(param["dW"], len(param["dW"]), len(param["dW"][0]), -self.lr)
            param["W"] = mat_add(param["W"], grad_step_w, len(grad_step_w), len(grad_step_w[0]), len(grad_step_w),
                                 len(grad_step_w[0]))
            grad_step_b = rescale(param["db"], len(param["db"]), len(param["db"][0]), -self.lr)
            param["b"] = mat_add(param["b"], grad_step_b, len(grad_step_b), len(grad_step_b[0]), len(grad_step_b),
                                 len(grad_step_b[0]))
