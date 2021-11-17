"""
Reference: https://cs231n.github.io/neural-networks-case-study/#grad
"""
import numpy_nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np


class MyNet(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden = nn.layers.Dense(in_features=self.input_dim,
                                      out_features=100,
                                      activation=nn.acts.ReLU(),
                                      weight_initializer=nn.inits.HeNormal(nn.acts.ReLU()),
                                      bias_initializer=nn.inits.Constant(0.),
                                      regularizer_type="l2",
                                      lam=1e-3
                                      )
        self.bn = nn.layers.BatchNorm1d(100)
        self.output = nn.layers.Dense(in_features=100,
                                      out_features=self.out_dim,
                                      weight_initializer=nn.inits.XavierUniform(),
                                      bias_initializer=nn.inits.Constant(0.),
                                      regularizer_type="l2",
                                      lam=1e-3
                                      )

    def forward(self, x, eval=False):
        x = self.hidden(x)
        x = self.bn(x, eval)
        return self.output(x)


np.random.seed(1)
random.seed(1)

num_samples = 100  # number of points per class
num_features = 2
num_classes = 3  # number of classes

epoch = 500
batch_size = 64
lr = 0.07

x = [[None for _ in range(num_features)] for _ in range(num_classes * num_samples)]
t = [[None] for _ in range(num_classes * num_samples)]

r = [i / num_samples for i in range(num_samples)]
for j in range(num_classes):
    theta = [i * 0.04 + j * 4 + random.gauss(0, 0.2) for i in range(num_samples)]
    for idx, radius, angle in zip(range(num_samples * j, num_samples * (j + 1)), r, theta):
        x[idx][0] = radius * np.sin(angle)
        x[idx][1] = radius * np.cos(angle)
        t[idx][0] = j

my_net = MyNet(num_features, num_classes)
ce_loss = nn.losses.CrossEntropyLoss()
opt = nn.optims.Adam(my_net.parameters, lr=lr)
loss_history = []
smoothed_loss = 0
for step in range(epoch):
    batch, target = [[None] for _ in range(batch_size)], [[None] for _ in range(batch_size)]
    for i in range(batch_size):
        idx = random.randint(0, len(x) - 1)
        batch[i] = x[idx]
        target[i] = t[idx]
    y = my_net(batch)
    loss = ce_loss(y, target)
    tot_loss = loss.value + \
               0.5 * my_net.hidden.lam * np.sum(my_net.hidden.vars["W"] ** 2) + \
               0.5 * my_net.output.lam * np.sum(my_net.output.vars["W"] ** 2)
    if step == 0:
        smoothed_loss = tot_loss
    else:
        smoothed_loss = 0.9 * smoothed_loss + 0.1 * tot_loss
    loss_history.append(smoothed_loss)
    my_net.backward(loss)
    opt.apply()
    if step % 10 == 0:
        print("Step: %i | loss: %.5f" % (step, tot_loss))

y = my_net.forward(x, eval=True)
predicted_class = np.argmax(y, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == np.array(t).squeeze(-1))))
plt.plot(np.arange(len(loss_history)), loss_history)
plt.figure()
plt.subplot(121)
plt.title("predicted classes")
plt.scatter(np.array(x)[:, 0], np.array(x)[:, 1], c=np.array(predicted_class), s=40)  # , cmap=plt.cm.Spectral)
plt.subplot(122)
plt.title("real classes")
plt.scatter(np.array(x)[:, 0], np.array(x)[:, 1], c=np.array(t), s=40)  # , cmap=plt.cm.Spectral)
plt.show()
