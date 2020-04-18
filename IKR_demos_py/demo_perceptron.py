import matplotlib.pyplot as plt
import numpy as np

red_data  = np.array([[0.2, 0.4, 0.5, 0.7, 0.9], [0.7, -0.5, 0.3, -0.9, 0.9]]).T
blue_data = np.array([[-0.1, -0.4, -0.5, -0.6, -0.7], [-0.2, 0.7, 0.6, 0.9, 0.5]]).T

plt.plot(red_data[:,0], red_data[:,1], 'r.')
plt.plot(blue_data[:,0], blue_data[:,1], 'b.')
plt.axis([-1, 1, -1, 1])
plt.show()

w = np.array([-0.1867, 0.7258])
data = np.r_[red_data, blue_data]
labs = np.r_[np.ones(len(red_data)), -np.ones(len(blue_data))]

solved = False
while not solved:
    solved = True
    for x, t in zip(data, labs):
        score = x.dot(w) * t
        if score < 0:
            solved = False
            plt.plot(x[0], x[1], 'go', ms=12, lw=2)
            plt.plot(red_data[:,0],  red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
            plt.plot([0, w[0]], [0, w[1]], 'k')
            plt.plot([-w[1] * 10, w[1] * 10], [w[0] * 10, -w[0] * 10], 'k')
            plt.axis([-1, 1, -1, 1])
            plt.show()
            w = w + x*t

plt.plot(red_data[:,0],  red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
plt.plot([0, w[0]], [0, w[1]], 'k')
plt.plot([-w[1] * 10, w[1] * 10], [w[0] * 10, -w[0] * 10], 'k')
plt.axis([-1, 1, -1, 1])
plt.show()
