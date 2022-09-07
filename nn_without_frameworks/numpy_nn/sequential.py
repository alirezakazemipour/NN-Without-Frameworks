from .losses import Loss
from .layers import ParamLayer
from tabulate import tabulate
from copy import deepcopy as dc


class Sequential:
    def __init__(self, *args):
        self._layers = args
        self._parameters = {i: self._layers[i].vars for i in range(len(self._layers)) if
                            isinstance(self._layers[i], ParamLayer)
                            }
        self._has_been_built = False

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
        delta = dict(delta=loss.delta)
        for layer in self._layers[::-1]:
            delta = layer.backward(**delta)

    def build(self, batch):
        self(batch)
        self._has_been_built = True

    def summary(self):
        if not self._has_been_built:
            raise Exception(f"You should first call the build method or perform a feedforward pass"
                            f" before invoking summary for {self.__class__.__name__}!"
                            )
        print("\nModel Summary:")
        data = []
        if isinstance(self._layers[0].input_shape, tuple):
            input_shape = (-1,) + self._layers[0].input_shape
        else:
            input_shape = -1, self._layers[0].input_shape
        name, output_shape, n_param = "Input", input_shape, 0
        data.append((name, output_shape, n_param))
        for i, layer in enumerate(self._layers):
            name, output_shape, n_param = layer.summary()
            name += f"[{i}]"
            data.append((name, output_shape, n_param))

        total_param = 0
        for x in data:
            *_, n_param = x
            total_param += n_param

        print(tabulate(data, headers=["Layer", "Output shape", "Param#"], tablefmt="grid"))
        print(f"total trainable parameters: {total_param}\n")

    def set_weights(self, params):
        self._parameters = dc(params)
        for i, layer in enumerate(self._layers):
            if isinstance(self._layers[i], ParamLayer):
                self._layers[i].vars = self._parameters[i]
