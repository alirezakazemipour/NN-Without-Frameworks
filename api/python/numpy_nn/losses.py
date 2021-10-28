import numpy as np

from abc import ABC


def supported_losses():
    return [x.__name__ for x in LossFunc.__subclasses__()]


class Loss:
    def __init__(self, value, delta):
        self.value = value
        self.delta = delta


class LossFunc:
    def __init__(self, pred=None, target=None):
        self.pred = pred
        self.target = target

    def apply(self, p, t):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def __call__(self, p, t):
        return self.apply(p, t)


class MSELoss(LossFunc, ABC):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def apply(self, p, t):
        super(MSELoss, self).__init__(p, t)
        return Loss(np.mean((p - t) ** 2) / 2, self.delta)

    @property
    def delta(self):
        return self.pred - self.target

