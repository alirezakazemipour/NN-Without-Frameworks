import numpy as np

from .module import Module
from .sequential import Sequential
from . import activations as acts
from . import initializers as inits
from . import layers, losses
from . import optimizers as optims


def seed(seed):
    np.random.seed(seed)
