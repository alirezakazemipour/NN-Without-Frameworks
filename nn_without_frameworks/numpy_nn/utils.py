import numpy as np
import inspect
from typing import Callable


def check_shapes(func):
    """Decorate 2-input-argument functions that their arguments must have the same numpy shape.

    It's very common in calculations involved in Neural Networks' implementation and learning loops
    to compute quantities that are based on 2 separate inputs and those inputs must possess the same shape, if one
    is hoping to obtain correct outputs.

    This function acts as a decorator and guarantees that any pair of inputs to functions decorated by it,
    does not violate the presumption of having equal shapes.

    Parameters
    ----------
    func : Callable
        The function to be decorated

    Returns
    -------
    Callable
        The decorated function.

    Notes
    -----
    Class methods, in addition to regular functions, are also supported. In other words,
    you can decorate class methods that one of the arguments is `self` and, this decorator
    will ignore `self` and performs the dimensionality checking on the remianing two arguments.

    Examples
    --------
    **Regular functions**

    >>> @check_shapes
    >>> def binary_cross_entropy(p, t):
    ...     eps = 1e-6
    ...     return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)
    >>> t = np.array([[1, 0, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> binary_cross_entropy(p, t)
    AssertionError: Inputs to the function are in different shapes: (1, 4) and (1, 3) at binary_cross_entropy!

    **Class methods**

    >>> class DummyClass:
    ...    @check_shapes
    ...    def binary_cross_entropy(self, p, t):
    ...        eps = 1e-6
    ...        return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)
    >>> dummy_var = DummyClass()
    >>> t = np.array([[1, 0, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> dummy_var.binary_cross_entropy(p, t)
    AssertionError: Inputs to the function are in different shapes: (1, 4) and (1, 3) at DummyClass.binary_cross_entropy!
    """
    if "self" in inspect.signature(func).parameters:
        def inner_func(self, x, y):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"
            return func(self, x, y)
    else:
        def inner_func(x, y):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"

            return func(x, y)

    return inner_func


@check_shapes
def binary_cross_entropy(p, t):
    r"""Calculate Binary Cross Entropy.

    When Binary Cross Entropy quantity is needed, this function can be invoked as a wrapper of calculating BCE.

    Parameters
    ----------
    p : array_like
        Prediction probabilities.
    t : array_like
        Target labels.

    Returns
    -------
    numpy.ndarray
        BCE values.

    Raises
    ------
    AssertionError
        If `p` and `t` shapes are not the same.

    Notes
    -----
    Binary Cross Entropy is a special case of cross entropy quantity that is concerned for only 2 categories:

    .. math:: BCE(p, t) = t * \log{(p)} + (1 - t) * \log{(1 - p)}

    `eps=1e-6` is used to stablized the logrithm function when it encounters 0 as its input.

    Examples
    --------
    >>> t = np.array([[1, 0, 1, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> binary_cross_entropy(p, t)
    [[-0.16251775 -0.2231423  -0.69314518 -0.05129224]]
    """
    eps = 1e-6
    return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)


def im2col_indices(x, kernel_size, stride, padding):
    """
    ``References:
    - [numpy-ml](https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L486)
    - [Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)
    - [Demystifying the math and implementation of Convolutions: Part III](https://praisethemoon.org/demystifying-the-math-and-implementation-of-convolutions-part-iii/)
    """
    fr, fc = kernel_size
    pr, pc = padding
    batch_size, in_rows, in_cols, in_channel = x.shape

    out_rows = conv_shape(in_rows, fr, stride, pr)
    out_cols = conv_shape(in_cols, fc, stride, pc)

    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, in_channel)
    i1 = stride * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * in_channel)
    j1 = stride * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (out_height * out_width, k^2C)
    # j.shape = (out_height * out_width, k^2C)
    # k.shape = (1, k^2C)
    i = i0.reshape(1, -1) + i1.reshape(-1, 1)
    j = j0.reshape(1, -1) + j1.reshape(-1, 1)
    k = np.repeat(np.arange(in_channel), fr * fc).reshape(1, -1)
    return i, j, k


def col2im(x_col, i, j, k, batch_size, n_rows, n_cols, n_channel, kernel_size, padding):
    pr, pc = padding
    fr, fc = kernel_size
    x_pad = np.zeros((batch_size, n_rows + 2 * pr, n_cols + 2 * pc, n_channel))

    x_col_reshaped = x_col.reshape(-1, n_channel * fr * fc, batch_size)
    x_col_reshaped = x_col_reshaped.transpose(2, 0, 1)

    np.add.at(x_pad, (slice(None), i, j, k), x_col_reshaped)

    pr2 = None if pr == 0 else -pr
    pc2 = None if pc == 0 else -pc
    return x_pad[:, pr:pr2, pc:pc2, :]


def conv_shape(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2 * padding - kernel_size) // stride + 1


if __name__ == "__main__":
    class DummyClass:
        @check_shapes
        def binary_cross_entropy(self, p, t):
            ...
            eps = 1e-6
            ...
            return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)


    dummy_var = DummyClass()
    t = np.array([[1, 0, 1]])
    p = np.array([[0.85, 0.2, 0.5, 0.95]])
    dummy_var.binary_cross_entropy(p, t)
