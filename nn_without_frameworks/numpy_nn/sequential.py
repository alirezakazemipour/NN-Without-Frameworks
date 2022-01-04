from .losses import Loss
from .layers import Layer


class Sequential:
    def __init__(self, *args):
        self._layers = args
        self._parameters = {i: self._layers[i].vars for i in range(len(self._layers))}

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        for layer in self._layers:
            x = layer.forward(x, eval)
        return x

    @property
    def parameters(self):
        return self._parameters

    def backward(self, loss):
        assert isinstance(loss, Loss)
        delta = loss.delta
        for layer in self._layers[::-1]:
            delta = layer.backward(delta)
