import matplotlib.pyplot as plt
from ikrlib import rand_gauss, plot2dfun, gellipse, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm, logistic_sigmoid
import numpy as np
from numpy.random import randint


#Generate random data for classes X1 and X2. The data for each class are
#generated from two gaussian distributions. Hopefully, we will be able to
#learn these distributions from data using EM algorithm implemented in 
#'train_gmm' function.
x1 = np.r_[rand_gauss(400, np.array([50, 40]), np.array([[100, 70], [70, 100]])),
           rand_gauss(200, np.array([55, 75]), np.array([[25, 0], [0, 25]]))]
          
x2 = np.r_[rand_gauss(400, np.array([45, 60]), np.array([[40, 0], [0, 40]])),
           rand_gauss(200, np.array([30, 40]), np.array([[20, 0], [0, 40]]))]
          
mu1, cov1 = train_gauss(x1)
mu2, cov2 = train_gauss(x2)
p1 = p2 = 0.5

# Plot the data
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
gellipse(mu1, cov1, 100, 'r')
gellipse(mu2, cov2, 100, 'b')
ax = plt.axis()
plt.show()

# Define functions which takes data as a parameter and, for our models of the two classes, ...
# make hard decision - return one to decide for calss X1 and zero otherwise
hard_decision = lambda x: logpdf_gauss(x, mu1, cov1) + np.log(p1) > logpdf_gauss(x, mu2, cov2) + np.log(p2)

# compute posterior probability for class X1
x1_posterior  = lambda x: logistic_sigmoid(logpdf_gauss(x, mu1, cov1) + np.log(p1) - logpdf_gauss(x, mu2, cov2) - np.log(p2))

# Plot the data with the hard decision as the background
plot2dfun(hard_decision, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
gellipse(mu1, cov1, 100, 'r')
gellipse(mu2, cov2, 100, 'b')

# Plot the data with the posterior probability as the background
plt.figure()
plot2dfun(x1_posterior, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
gellipse(mu1, cov1, 100, 'r')
gellipse(mu2, cov2, 100, 'b')
plt.show()

# Train and test with GMM models with full covariance matrices
#Decide for number of gaussian mixture components used for the model
m1 = 2

# Initialize mean vectors to randomly selected data points from corresponding class
mus1 = x1[randint(1, len(x1), m1)]

# Initialize all covariance matrices to the same covariance matrices computed using
# all the data from the given class
covs1 = [cov1] * m1

# Use uniform distribution as initial guess for the weights
ws1 = np.ones(m1) / m1

m2 = 2
mus2 = x2[randint(1, len(x2), m2)]
covs2 = [cov2] * m2
ws2 = np.ones(m2) / m2

#fig = plt.figure()
#ims = []

# Run 30 iterations of EM algorithm to train the two GMM models
for i in range(30):
    plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 100, 'r', lw=round(w * 10))
    for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 100, 'b', lw=round(w * 10))
    ws1, mus1, covs1, ttl1 = train_gmm(x1, ws1, mus1, covs1)
    ws2, mus2, covs2, ttl2 = train_gmm(x2, ws2, mus2, covs2)
    print('Total log-likelihood: %s for class X1; %s for class X2' % (ttl1, ttl2))
    plt.show()

hard_decision = lambda x: logpdf_gmm(x, ws1, mus1, covs1) + np.log(p1) > logpdf_gmm(x, ws2, mus2, covs2) + np.log(p2)
plot2dfun(hard_decision, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.')
plt.plot(x2[:,0], x2[:,1], 'b.')
for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 100, 'r', lw=round(w * 10))
for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 100, 'b', lw=round(w * 10))

plt.figure()
x1_posterior  = lambda x: logistic_sigmoid(logpdf_gmm(x, ws1, mus1, covs1) + np.log(p1) - logpdf_gmm(x, ws2, mus2, covs2) - np.log(p2))
plot2dfun(x1_posterior, ax, 500)
plt.plot(x1[:,0], x1[:,1], 'r.')
plt.plot(x2[:,0], x2[:,1], 'b.')
for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 100, 'r', lw=round(w * 10))
for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 100, 'b', lw=round(w * 10))
plt.show()
