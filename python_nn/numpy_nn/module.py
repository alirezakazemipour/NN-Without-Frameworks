from .losses import Loss
from .layers import Layer


class Module:
    def __init__(self):
        self._parameters = {}
        self._layers = []

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        raise NotImplementedError

    @property
    def parameters(self):
        return self._parameters

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            layer = value
            self._parameters[key] = layer.vars
            self._layers.append(value)
        object.__setattr__(self, key, value)

    def backward(self, loss):
        assert isinstance(loss, Loss)
        delta = loss.delta
        for layer in self._layers[::-1]:
            delta = layer.backward(delta)
