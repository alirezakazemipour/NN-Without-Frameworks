"""
Reference: https://cs231n.github.io/neural-networks-case-study/#grad
"""
import python_nn.numpy_nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np


class MyNet(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden1 = nn.layers.Dense(in_features=self.input_dim,
                                       out_features=100,
                                       activation=nn.acts.ReLU(),
                                       weight_initializer=nn.inits.HeNormal(nn.acts.ReLU()),
                                       bias_initializer=nn.inits.Constant(0.)
                                       )

        self.output = nn.layers.Dense(in_features=100,
                                      out_features=self.out_dim,
                                      weight_initializer=nn.inits.XavierUniform(),
                                      bias_initializer=nn.inits.Constant(0.))

    def forward(self, x):
        x = self.hidden1(x)
        return self.output(x)


np.random.seed(1)
random.seed(1)

num_samples = 100  # number of points per class
num_features = 2
num_classes = 3  # number of classes

x = [[None for _ in range(num_features)] for _ in range(num_classes * num_samples)]
t = [[None] for _ in range(num_classes * num_samples)]

r = [i / num_samples for i in range(num_samples)]
for j in range(num_classes):
    theta = [i * 0.04 + j * 4 + random.gauss(0, 0.2) for i in range(num_samples)]
    for idx, radius, angle in zip(range(num_samples * j, num_samples * (j + 1)), r, theta):
        x[idx][0] = radius * np.sin(angle)
        x[idx][1] = radius * np.cos(angle)
        t[idx][0] = j

# plt.scatter(np.array(x)[:, 0], np.array(x)[:, 1], c=np.array(t), s=40, cmap=plt.cm.Spectral)
# plt.show()

epoch = 4000
batch_size = 64

my_net = MyNet(num_features, num_classes)
ce_loss = nn.losses.CrossEntropyLoss()
opt = nn.optims.SGD(my_net.parameters, lr=1.)
loss_history = []
for step in range(epoch):
    batch, target = [[None] for _ in range(batch_size)], [[None] for _ in range(batch_size)]
    for i in range(batch_size):
        idx = random.randint(0, len(x) - 1)
        batch[i] = x[idx]
        target[i] = t[idx]
    y = my_net(batch)
    loss = ce_loss(y, target)
    loss_history.append(loss.value)
    my_net.backward(loss)
    opt.apply()
    if step % 5 == 0:
        print("Step: %i | loss: %.5f" % (step, loss.value))

# plt.scatter(x, t, s=20)
y = my_net.forward(x)
# plt.plot(x, y, c="red", lw=3)
# plt.show()
predicted_class = np.argmax(y, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == np.array(t).squeeze(-1))))
plt.plot(np.arange(len(loss_history)), loss_history)
plt.show()
