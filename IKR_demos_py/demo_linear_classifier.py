import matplotlib.pyplot as plt
from ikrlib import rand_gauss, train_generative_linear_classifier, plot2dfun, gellipse
from ikrlib import logistic_sigmoid, train_linear_logistic_regression, train_linear_logistic_regression_GD
import numpy as np
#from numpy.random import randn


red_data = rand_gauss(1000, np.array([50, 50]), np.array([[100,  70], 
                                                          [ 70, 100]]))
blue_data = rand_gauss(1000, np.array([40, 70]), np.array([[40,  0],
                                                           [ 0, 40]]))


#With blue_data, Generative Linear Classifier fails to perform well
hovado = rand_gauss(80, np.array([-20, 110]), np.array([[20, 0], 
                                                        [ 0, 20]]))                                                        
blue_data = np.r_[blue_data, hovado]

x = np.r_[red_data, blue_data]
t = np.r_[np.ones(len(red_data)), np.zeros(len(blue_data))]

plt.plot(red_data[:,0], red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
ax = plt.axis()
plt.show()

w, w0, data_cov = train_generative_linear_classifier(x, t)
x1, x2 = ax[:2]
y1 = (-w0 - (w[0] * x1)) / w[1]
y2 = (-w0 - (w[0] * x2)) / w[1]

plt.plot(red_data[:,0], red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
plt.plot([x1, x2], [y1, y2], 'k', linewidth=2)
gellipse(np.mean(red_data, axis=0), data_cov, 100, 'r')
gellipse(np.mean(blue_data, axis=0), data_cov, 100, 'r')
plt.show()

for i in range(100):
    plot2dfun(lambda x: logistic_sigmoid(x.dot(w) + w0), ax, 1000)
    plt.plot(red_data[:,0], red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
    plt.show()
    w, w0 = train_linear_logistic_regression(x, t, w, w0)
