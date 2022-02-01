import numpy as np
import inspect


def binary_cross_entropy(p, t):
    return t * np.log(p + 1e-6) + (1 - t) * np.log(1 - p + 1e-6)


def check_shapes(func):
    if inspect.signature(func).parameters:
        def inner_func(self, x, y):
            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"
            return func(self, x, y)
    else:
        def inner_func(x, y):
            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"

            return func(x, y)

    return inner_func


if __name__ == "__main__":
    print(type(binary_cross_entropy))
