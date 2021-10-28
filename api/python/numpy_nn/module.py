import api.python.numpy_nn as nn


class Module:
    def __init__(self):
        self._parameters = {}
        self._layers = []

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    @property
    def parameters(self):
        return self._parameters

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.Layer):
            layer = value
            self._parameters[key] = layer.vars
            self._layers.append(value)
        object.__setattr__(self, key, value)

    def backward(self, loss):
        assert isinstance(loss, nn.losses.Loss)
        delta = loss.delta
        for name, layer in zip(list(self._parameters.keys())[::-1], self._layers[::-1]):
            delta = layer.backward(delta)
