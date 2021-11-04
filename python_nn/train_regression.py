import python_nn.pure_nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = nn.layers.Dense(in_features=self.input_dim,
                                       out_features=10,
                                       activation=nn.acts.ReLU(),
                                       weight_initializer=nn.inits.RandomUniform(),
                                       bias_initializer=nn.inits.Constant(0.)
                                       )

        self.output = nn.layers.Dense(in_features=10,
                                      out_features=1,
                                      weight_initializer=nn.inits.RandomUniform(),
                                      bias_initializer=nn.inits.Constant(0.))

    def forward(self, x):
        x = self.hidden1(x)
        return self.output(x)


np.random.seed(1)
random.seed(1025)
x = [[0.01 * i] for i in range(-100, 100)]
t = [[k[0] ** 2 + random.gauss(0, 1) * 0.1] for k in x]

my_net = MyNet(1)
mse = nn.losses.MSELoss()
opt = nn.optims.AdaGrad(my_net.parameters, lr=0.1)
# opt = nn.optims.RMSProp(my_net.parameters, lr=0.001, beta=0.99)
# opt = nn.optims.Momentum(my_net.parameters, lr=0.3, mu=0.8)
# opt = nn.optims.SGD(my_net.parameters, lr=0.3)
loss_history = []
for epoch in range(1000):
    y = my_net(x)
    loss = mse(y, t)
    loss_history.append(loss.value)
    my_net.backward(loss)
    opt.apply()
    print("Step: %i | loss: %.5f" % (epoch, loss.value))

plt.scatter(x, t, s=20)
plt.plot(x, y, c="red", lw=3)
# plt.show()
# plt.plot(np.arange(len(loss_history)), loss_history)
plt.show()