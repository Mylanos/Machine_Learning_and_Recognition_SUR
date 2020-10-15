import matplotlib.pyplot as plt
from ikrlib import k_nearest_neighbours, gellipse, rand_gauss, plot2dfun
import numpy as np

x1 = rand_gauss(100, np.array([50, 50]), np.array([[100, 70], [70, 100]]))
x2 = rand_gauss(100, np.array([40, 60]), np.array([[40, 0], [0, 40]]))
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
ax = plt.axis()
plt.show()

k = 9

def soft_score(x):
  return k_nearest_neighbours(x, x1, x2, k)

def hard_decision(x):
  return soft_score(x) > 0.5

plot2dfun(hard_decision, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
plt.show()

plot2dfun(soft_score, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
plt.show()
