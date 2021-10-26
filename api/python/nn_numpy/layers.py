from abc import ABC
import numpy as np
import api.python.nn_numpy as nn


class Layer:
    def __init__(self):
        self.vars = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer):
        super().__init__()

        self.w = weight_initializer.initialize(weight_shape)
        self.b = bias_initializer.initialize(weight_shape[1])

        self.vars["W"] = self.w
        self.vars["b"] = self.b


class Dense(ParamLayer, ABC):
    def __init__(self, in_features: int,
                 out_features: int,
                 activation: nn.acts = nn.acts.Linear(),
                 weight_initializer: nn.inits = nn.inits.RandomUniform(),
                 bias_initializer: nn.inits = nn.inits.Constant()
                 ):
        super().__init__(weight_shape=(in_features, out_features),
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer
                         )
        self.in_features = in_features
        self.out_features = out_features
        self.act = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        z = x.dot(self.w) + self.b
        a = self.act(z)
        return a

    def backward(self, x):
        
    def __call__(self, x):
        return self.forward(x)