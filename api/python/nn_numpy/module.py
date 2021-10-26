import api.python.nn_numpy as nn
import numpy as np


class Module:
    def __init__(self):
        self._parameters = {}

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.Layer):
            layer = value
            self._parameters[key] = {
                "vars": layer.vars,
                "grads": {k: np.zeros_like(k) for k in layer.vars.keys()}
            }
        object.__setattr__(self, key, value)
