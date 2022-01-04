from .initializers import *
from .activations import *


def supported_layers():
    return [x.__name__ for x in ParamLayer.__subclasses__()]


class Layer:
    def __init__(self):
        self.vars = {}
        self._input_shape = None

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1])
        return name, output_shape, n_param

    @property
    def input_shape(self):
        return self.vars["W"].shape[0]

    def forward(self, x, eval=False):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer,
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        super().__init__()

        self.vars["W"] = weight_initializer.initialize(weight_shape)
        self.vars["b"] = bias_initializer.initialize((1, weight_shape[1]))
        self.vars["dW"] = np.zeros(weight_shape)
        self.vars["db"] = np.zeros((1, weight_shape[1]))

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

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        z = x.dot(self.vars["W"]) + self.vars["b"]
        self.z = z
        a = self.act(z)
        return a

    def backward(self, delta):
        #  https://cs182sp21.github.io/static/slides/lec-5.pdf
        dz = delta * self.act.derivative(self.z)
        self.vars["dW"] = self.input.T.dot(dz) / dz.shape[0]
        self.vars["db"] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.regularizer_type == "l2":
            self.vars["dW"] += self.lam * self.vars["W"]
            # self.vars["db"] += self.lam * self.vars["b"]
        elif self.regularizer_type == "l1":
            self.vars["dW"] += self.lam
            # self.vars["db"] += self.lam

        delta = dz.dot(self.vars["W"].T)
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

    def backward(self, delta):
        #  https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dz = delta
        dx_hat = dz * self.vars["W"]
        m = dz.shape[0]
        self.vars["dW"] = np.sum(self.x_hat * dz, axis=0) / m
        self.vars["db"] = np.sum(dz, axis=0) / m

        delta = (m * dx_hat - np.sum(dx_hat, axis=0, keepdims=True) - self.x_hat * np.sum(
            dx_hat * self.x_hat, axis=0, keepdims=True)) / (m * np.sqrt(self.std ** 2 + self.eps))
        return delta

    def __call__(self, x, eval=False):
        return self.forward(x, eval)
