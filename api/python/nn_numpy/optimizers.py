from abc import ABC


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.vars = []
        self.grads = []
        for param in self.params.values():
            for k in param.keys():
                self.vars.append(param["vars"][k])
                self.grads.append(param["grads"][k])

    def apply(self):
        raise NotImplementedError


class SGD(Optimizer, ABC):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def apply(self):
        for var, grad in zip(self.vars, self.grads):
            var -= self.lr * grad
