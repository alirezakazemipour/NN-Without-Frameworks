from .losses import Loss
from tabulate import tabulate


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

    def summary(self):
        print("\nModel Summary:")
        data = []
        name, output_shape, n_param = "Input", (None, self._layers[0].input_shape), 0
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
