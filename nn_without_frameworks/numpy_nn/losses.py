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


class CrossEntropyLoss(LossFunc, ABC):
    #  https://cs231n.github.io/neural-networks-case-study/#grad
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def apply(self, p, t):
        super(CrossEntropyLoss, self).__init__(p, t)
        probs = self.soft_max(p)
        loss = -np.log(probs[range(p.shape[0]), np.array(t).squeeze(-1)])

        return Loss(np.mean(loss), self.delta)

    @property
    def delta(self):
        #  https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
        probs = self.soft_max(self.pred)
        probs[range(self.pred.shape[0]), np.array(self.target).squeeze(-1)] -= 1

        return probs

    @staticmethod
    def soft_max(x):
        logits = x - np.max(x, axis=-1, keepdims=True)
        num = np.exp(logits)
        den = np.sum(num, axis=-1, keepdims=True)
        return num / den
