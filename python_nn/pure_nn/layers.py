from abc import ABC
import python_nn.numpy_nn as nn
from .utils import mat_mul, mat_add, element_wise_mul, transpose, rescale
from copy import deepcopy


def supported_layers():
    return [x.__name__ for x in ParamLayer.__subclasses__()]


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

        i, j = weight_shape
        self.vars["W"] = weight_initializer.initialize([[0 for _ in range(j)] for _ in range(i)])
        self.vars["b"] = [bias_initializer.initialize([0 for _ in range(j)])]
        self.vars["dW"] = [[0 for _ in range(j)] for _ in range(i)]
        self.vars["db"] = [[0 for _ in range(j)]]

        self.z = None
        self.input = None


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
        assert isinstance(x, list)
        assert isinstance(x[0], list), "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        # z = x.dot(self.vars["W"]) + self.vars["b"]
        z = mat_mul(x, self.vars["W"])
        b = deepcopy(self.vars["b"])
        while len(b) < len(x):
            b.append(self.vars["b"][0])
        z = mat_add(z, b)
        self.z = z
        a = self.act(z)
        return a

    def backward(self, delta):
        dz = element_wise_mul(delta, self.act.derivative(self.z))
        input_t = transpose(self.input)
        dw_unscale = mat_mul(input_t, dz)
        self.vars["dW"] = rescale(dw_unscale, 1 / len(dz))
        # self.vars["db"] = np.sum(dz, axis=0) / dz.shape[0]
        ones_t = [[1 for _ in range(len(dz))] for _ in range(1)]
        db_unscale = mat_mul(ones_t, dz)
        self.vars["db"] = rescale(db_unscale, 1 / len(dz))
        w_t = transpose(self.vars["W"])
        # delta = dz.dot(self.vars["W"].T)
        delta = mat_mul(dz, w_t)
        return delta

    def __call__(self, x):
        return self.forward(x)
