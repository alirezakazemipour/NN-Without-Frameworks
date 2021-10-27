from api.python import nn_numpy as nn
import numpy as np
import matplotlib.pyplot as plt


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = nn.layers.Dense(in_features=self.input_dim,
                                       out_features=10,
                                       activation=nn.acts.Tanh(),
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
x = np.linspace(-1, 1, 256)[:, None]  # [batch, 1]
t = x ** 2 + np.random.normal(0., 0.1, (256, 1))  # [batch, 1]

my_net = MyNet(1)
mse = nn.losses.MSELoss()
opt = nn.optims.SGD(my_net.parameters, 0.1)
for epoch in range(1000):
    y = my_net(x)
    loss = mse(y, t)
    my_net.backward(loss)
    opt.apply()
    print("Step: %i | loss: %.5f" % (epoch, loss.value))

plt.scatter(x, t, s=20)
plt.plot(x, y, c="red", lw=3)
plt.show()
