import numpy as np

from .initializers import *
from .activations import *


# TODO:
# __str__()
# __repr__()

def supported_layers():
    return [x.__name__ for x in ParamLayer.__subclasses__()]


class Layer:
    def forward(self, **kwargs):
        raise NotImplementedError

    def backward(self, **x):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer,
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        self.vars = {"W": weight_initializer.initialize(weight_shape),
                     "b": bias_initializer.initialize((1, weight_shape[1])),
                     "dW": np.zeros(weight_shape),
                     "db": np.zeros((1, weight_shape[1]))}

        self.z = None
        self.input = None

        self.regularizer_type = regularizer_type
        self.lam = lam

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1])
        return name, output_shape, n_param

    @property
    def input_shape(self):
        return self.vars["W"].shape[0]


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

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        z = x.dot(self.vars["W"]) + self.vars["b"]
        self.z = z
        a = self.act(z)
        return a

    def backward(self, **delta):
        #  https://cs182sp21.github.io/static/slides/lec-5.pdf
        delta = delta["delta"]
        dz = delta * self.act.derivative(self.z)
        self.vars["dW"] = self.input.T.dot(dz) / dz.shape[0]
        self.vars["db"] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.regularizer_type == "l2":
            self.vars["dW"] += self.lam * self.vars["W"]
            # Biases are not regularized: https://cs231n.github.io/neural-networks-2/#reg
            # self.vars["db"] += self.lam * self.vars["b"]
        elif self.regularizer_type == "l1":
            self.vars["dW"] += self.lam
            # self.vars["db"] += self.lam

        delta = dz.dot(self.vars["W"].T)
        return dict(delta=delta)

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


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
        self.mu_hat = 0
        self.std_hat = 0

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        if not eval:
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.std = np.std(x, axis=0, keepdims=True)
            self.mu_hat = (1 - self.beta) * self.mu_hat + self.beta * self.mu
            self.std_hat = (1 - self.beta) * self.std_hat + self.beta * self.std
        else:
            self.mu = self.mu_hat
            self.std = self.std_hat
        x_hat = x - self.mu / np.sqrt(self.std ** 2 + self.eps)
        self.x_hat = x_hat
        y = self.vars["W"] * x_hat + self.vars["b"]
        return y

    def backward(self, **delta):
        #  https://kevinzakka.github.io/2016/09/14/batch_normalization/
        delta = delta["delta"]
        dz = delta
        dx_hat = dz * self.vars["W"]
        m = dz.shape[0]
        self.vars["dW"] = np.sum(self.x_hat * dz, axis=0) / m
        self.vars["db"] = np.sum(dz, axis=0) / m

        delta = (m * dx_hat - np.sum(dx_hat, axis=0, keepdims=True) - self.x_hat * np.sum(
            dx_hat * self.x_hat, axis=0, keepdims=True)) / (m * np.sqrt(self.std ** 2 + self.eps))
        return dict(delta=delta)

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


class Dropout(Layer, ABC):
    """
    - References:
        1. https://cs231n.github.io/neural-networks-2/#reg
        2. https://deepnotes.io/dropout
    """

    def __init__(self, p: float = 0.5):
        """
        :param p: float
        Probability of keeping a neuron active.
        """

        self.p = p
        self.mask = None

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        if not eval:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            return x * self.mask
        else:
            return x

    def backward(self, **delta):
        delta = delta["delta"]
        return dict(delta=delta * self.mask)

    def summary(self):
        name = self.__class__.__name__
        return name, 0

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


class LSTMCell(ParamLayer, ABC):
    """
    - References:
        1. https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/layers/layers.py#L3857
        2. http://arunmallya.github.io/writeups/nn/lstm/index.html#/7
        3. http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture10.pdf
    """

    def __init__(self, in_features: int,
                 hidden_size: int,
                 weight_initializer: Initializer = RandomUniform(),
                 bias_initializer: Initializer = Constant(),
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        weight_shape = (in_features + hidden_size, 4 * hidden_size)
        super().__init__(weight_shape=weight_shape,
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lam=lam
                         )
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam

        self.c_t = None
        self.o = None
        self.g = None
        self.c = None
        self.i = None
        self.f = None
        self.g_hat = None

    def forward(self, x, h, c, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = np.hstack([x, h])
        self.c = c
        z = self.input.dot(self.vars["W"]) + self.vars["b"]
        i_hat, f_hat, o_hat, self.g_hat = np.split(z, 4, axis=-1)
        self.i = Sigmoid.forward(i_hat)
        self.f = Sigmoid.forward(f_hat)
        self.o = Sigmoid.forward(o_hat)
        self.g = np.tanh(self.g_hat)
        self.c_t = self.f * c + self.i * self.g
        h_t = self.o * np.tanh(self.c_t)
        return h_t, self.c_t

    def backward(self, **delta):
        dh_t = delta.get("h_t", delta["delta"])
        dc_t = delta.get("c_t", np.zeros_like(dh_t))
        do = dh_t * np.tanh(self.c_t)
        dc_t += dh_t * self.o * (1 - np.tanh(self.c_t) ** 2)
        di = dc_t * self.g
        df = dc_t * self.c
        dg = dc_t * self.i
        dc = dc_t * self.f
        dg_hat = dg * (1 - np.tanh(self.g_hat) ** 2)
        di_hat = di * self.i * (1 - self.i)
        df_hat = df * self.f * (1 - self.f)
        do_hat = do * self.o * (1 - self.o)
        dz = np.hstack([di_hat, df_hat, do_hat, dg_hat])

        self.vars["dW"] = self.input.T.dot(dz) / dz.shape[0]
        self.vars["db"] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        dinput = dz.dot(self.vars["W"].T)
        dx, dh = np.split(dinput, [self.in_features], axis=-1)

        return dict(delta=dx, h_t=dh, c_t=dc)

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1] // 4)
        return name, output_shape, n_param

    def __call__(self, x, h, c, eval=False):
        return self.forward(x, h, c, eval)
