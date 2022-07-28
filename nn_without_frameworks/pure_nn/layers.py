import numpy as np

from .utils import *
from .activations import *
from .initializers import *


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
                 bias_initializer,
                 regularizer_type: str = None,
                 lam: float = 0.):
        super().__init__()

        i, j = weight_shape
        init_weight = Matrix(i, j)
        init_bias = Matrix(1, j)
        self.vars["W"] = weight_initializer.initialize(init_weight)
        self.vars["b"] = bias_initializer.initialize(init_bias)
        self.vars["dW"] = Matrix([[0 for _ in range(j)] for _ in range(i)])
        self.vars["db"] = Matrix([[0 for _ in range(j)]])

        self.z = None
        self.input = None

        self.regularizer_type = regularizer_type
        self.lam = lam


class Dense(ParamLayer, ABC):
    def __init__(self, in_features: int,
                 out_features: int,
                 activation: Activation = Linear(),
                 weight_initializer: Initializer = RandomUniform(),
                 bias_initializer: Initializer = Constant(),
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        super().__init__(weight_shape=(in_features, out_features),
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lam=lam
                         )
        self.in_features = in_features
        self.out_features = out_features
        self.act = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = np.ndarray.tolist(x)
        if not isinstance(x, Matrix):
            x = Matrix(x)
        self.input = x
        # z = x.dot(self.vars["W"]) + self.vars["b"]
        z = x @ self.vars["W"]
        b = deepcopy(self.vars["b"])
        while len(b) < len(x):
            b.append(self.vars["b"][0])
        z = z + b
        self.z = z
        a = self.act(z)
        return a.value

    def backward(self, delta):
        dz = delta * self.act.derivative(self.z)
        dw_unscale = self.input.t() @ dz
        self.vars["dW"] = dw_unscale * (1 / len(dz))
        # self.vars["db"] = np.sum(dz, axis=0) / dz.shape[0]
        ones_t = Matrix([[1 for _ in range(len(dz))] for _ in range(1)])
        db_unscale = ones_t @ dz
        self.vars["db"] = db_unscale * (1 / len(dz))

        if self.regularizer_type == "l2":
            self.vars["dW"] = self.vars["dW"] + (self.vars["W"] * self.lam)
            # self.vars["db"] = mat_add(self.vars["db"], rescale(self.vars["b"], self.lam))

        elif self.regularizer_type == "l1":
            self.vars["dW"] = self.vars["dW"] + self.lam
            # self.vars["db"] = add_scalar(self.vars["db"], self.lam)

        # delta = dz.dot(self.vars["W"].T)
        delta = dz @ self.vars["W"].t()
        return delta

    def __call__(self, x):
        return self.forward(x)


class BatchNorm1d(ParamLayer, ABC):
    #  https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    def __init__(self, in_features: int):
        super().__init__(weight_shape=(1, in_features),
                         weight_initializer=Constant(1.),
                         bias_initializer=Constant(0.)
                         )
        self.in_features = in_features
        self.x_hat = None
        self.eps = 1e-5
        self.beta = 0.1
        self.mu = 0
        self.std = 0
        self.mu_hat = Matrix([[0 for _ in range(self.in_features)]])
        self.std_hat = Matrix([[1 for _ in range(self.in_features)]])
        self.gamma = None

    def forward(self, x: Matrix, eval=False):
        if not isinstance(x, Matrix):
            x = Matrix(x)
        if not eval:
            self.mu = x.mean()
            self.std = x.var() ** 0.5
            self.mu_hat = self.mu_hat * (1 - self.beta) + (self.mu * self.beta)
            self.std_hat = self.std_hat * (1 - self.beta) + (self.std * self.beta)
        else:
            self.mu = self.mu_hat
            self.std = self.std_hat
        mu = deepcopy(self.mu)
        std = deepcopy(self.std)
        while len(mu) < len(x):
            mu.append(self.mu[0])
            std.append(self.std[0])
        num = x + (mu * -1)
        den = ((std * std) + self.eps) ** 0.5
        x_hat = num * (den ** -1)
        self.x_hat = x_hat

        self.gamma = deepcopy(self.vars["W"])
        beta = deepcopy(self.vars["b"])
        while len(self.gamma) < len(x):
            self.gamma.append(self.vars["W"][0])
            beta.append(self.vars["b"][0])

        y = (self.gamma * x_hat) + beta
        return y.value

    def backward(self, delta: Matrix):
        #  https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dz = delta
        dx_hat = dz * self.gamma
        m = len(dz)
        self.vars["dW"] = (self.x_hat * dz).sum() * (1 / m)
        self.vars["db"] = dz.sum() * (1 / m)

        a1 = dx_hat * m
        a2 = dx_hat.sum()
        a3 = self.x_hat * (dx_hat * self.x_hat).sum()
        num = a1 + (a2 * -1) + (a3 * -1)
        den = (((self.std * self.std) + self.eps) ** 0.5) * m

        delta = num * (den ** -1)
        return delta

    def __call__(self, x, eval=False):
        return self.forward(x, eval)
