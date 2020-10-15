import matplotlib.pyplot as plt
from ikrlib import plot2dfun, eval_nnet, train_nnet
import numpy as np
from numpy.random import randn

n = 333
x1 = np.r_[randn(n, 2) + np.array([1, 3]),
           randn(n, 2) + np.array([-2, -2]),
           randn(n, 2) + np.array([0, 0])]
x2 = np.r_[randn(n, 2) + np.array([-2, 2]),
           randn(n, 2) + np.array([2, -2]),
           randn(n, 2) + np.array([4, 4]),
           randn(n, 2) + np.array([-4, -4])]

t1 = np.ones(len(x1))
t2 = np.zeros(len(x2))

mu = np.mean(np.r_[x1, x2], axis=0)
sig = np.std(np.r_[x1, x2], axis=0)

x1 = (x1 - mu) / sig
x2 = (x2 - mu) / sig

x = np.r_[x1, x2]
t = np.r_[t1, t2]

plt.plot(x1[:,0], x1[:,1], 'rx', x2[:,0], x2[:,1], 'bx')
ax = plt.axis()
plt.show()

dim_in = 2
dim_hidden = 5
dim_out = 1

w1 = randn(dim_in + 1, dim_hidden) * .1
w2 = randn(dim_hidden + 1, dim_out) * .1

epsilon = .05

for i in range(50):
    plot2dfun(lambda x: eval_nnet(x, w1, w2), ax, 100)
    plt.plot(x1[:,0], x1[:,1], 'rx', x2[:,0], x2[:,1], 'bx')
    plt.show()

    w1, w2, ed = train_nnet(x, t, w1, w2, epsilon)
    print('Total log-likelihood: %f' % -ed)
