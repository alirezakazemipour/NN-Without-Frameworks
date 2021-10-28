from abc import ABC
from .utils import mat_add, rescale


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
        # return Loss(np.mean((p - t) ** 2) / 2, self.delta)
        assert isinstance(p, list) and isinstance(t, list)
        assert isinstance(p[0], list) and isinstance(t[0], list), "target and prediction should be in batch mode: (batch_size, n_dims)"

        assert len(p) == len(t) and len(p[0]) == len(t[0])
        loss = 0
        for w, h in zip(p, t):
            for x, y in zip(w, h):
                loss += 0.5 * (x - y) ** 2
        return Loss(loss / len(p), self.delta)

    @property
    def delta(self):
        len_p = len(self.pred)
        len_t = len(self.target)
        return mat_add(self.pred, rescale(self.target, len_t, len(self.target[0]), -1), len_p, len(self.pred[0]), len_t, len(self.target[0]))

