import numpy as np

from api.python import nn_numpy as nn


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = nn.layers.Dense(in_features=self.input_dim,
                                      out_features=8,
                                      activation=nn.acts.ReLU(),
                                      weight_initializer=nn.inits.RandomUniform(),
                                      bias_initializer=nn.inits.Constant(0.)
                                      )
        self.output = nn.layers.Dense(in_features=8,
                                      out_features=1,
                                      weight_initializer=nn.inits.RandomUniform(),
                                      bias_initializer=nn.inits.Constant(0.))

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


my_net = MyNet(5)
input = np.ones((1, 5))
output = my_net(input)
print(input, output)
