import matplotlib.pyplot as plt
from ikrlib import raw8khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint

train_m = list(raw8khz2mfcc('gID_data/male/train').values())
train_f = list(raw8khz2mfcc('gID_data/female/train').values())
test_m  = list(raw8khz2mfcc('gID_data/male/test').values())
test_f  = list(raw8khz2mfcc('gID_data/female/test').values())

train_m = np.vstack(train_m)
train_f = np.vstack(train_f)
dim = train_m.shape[1]

# PCA reduction to 2 dimensions

cov_tot = np.cov(np.vstack([train_m, train_f]).T, bias=True)
# take just 2 largest eigenvalues and corresponding eigenvectors
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

train_m_pca = train_m.dot(e)
train_f_pca = train_f.dot(e)
plt.plot(train_m_pca[:,1], train_m_pca[:,0], 'b.', ms=1)
plt.plot(train_f_pca[:,1], train_f_pca[:,0], 'r.', ms=1)
plt.show()
# Classes are not well separated in 2D PCA subspace


# LDA reduction to 1 dimenzion (only one LDA dimension is available for 2 tridy)
n_m = len(train_m)
n_f = len(train_f)
cov_wc = (n_m*np.cov(train_m.T, bias=True) + n_f*np.cov(train_f.T, bias=True)) / (n_m + n_f)
cov_ac = cov_tot - cov_wc
d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))
plt.figure()
junk = plt.hist(train_m.dot(e), 40, histtype='step', color='b') #, normed=True
junk = plt.hist(train_f.dot(e), 40, histtype='step', color='r') # , normed=True
plt.show()
# Distribution in this single dimensional space are reasonable separated

# Lets define uniform a-priori probabilities of classes:
P_m = 0.5
P_f = 1 - P_m   

# For one male test utterance (test_m[0]), obtain frame-by-frame log-likelihoods
# with two models, one trained using male and second using feamle training data.
# In this case, the models are single gaussians with diagonal covariance matrices.

ll_m = logpdf_gauss(test_m[0], np.mean(train_m, axis=0), np.var(train_m, axis=0))
ll_f = logpdf_gauss(test_m[0], np.mean(train_f, axis=0), np.var(train_f, axis=0))

# Plot the frame-by-frame likelihoods obtained with the two models; Note that
# 'll_m' and 'll_f' are log likelihoods, so we need to use exp function
plt.figure() 
plt.plot(np.exp(ll_m), 'b')
plt.plot(np.exp(ll_f), 'r')
plt.show()

# Plot frame-by-frame posteriors
posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f)
# Alternatively the posterior can by computed using log odds ratio and logistic sigmoid function as:
# posterior_m = logistic_sigmoid(ll_m - ll_f + log(P_m/P_f));
plt.figure()
plt.plot(posterior_m, 'b')
plt.plot(1- posterior_m, 'r')
plt.show()

# Plot frame-by-frame log-likelihoods
plt.figure()
plt.plot(ll_m, 'b')
plt.plot(ll_f, 'r')
plt.show()

# But, we do not want to make frame-by-frame decisions. We want to recognize the
# whole segment. Aplying frame independeny assumption, we sum log-likelihoods.
# We decide for class 'male' if the following quantity is positive.
#print (sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f))


# Repeating the whole excercise, now with gaussian models with full covariance
# matrices

ll_m = logpdf_gauss(test_m[0], *train_gauss(train_m)) 
ll_f = logpdf_gauss(test_m[0], *train_gauss(train_f))
# '*' before 'train_gauss' pases both return values (mean and cov) as parameters of 'logpdf_gauss' 
posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f);
plt.figure(); plt.plot(posterior_m, 'b'); plt.plot(1-posterior_m, 'r');
plt.figure(); plt.plot(ll_m, 'b');        plt.plot(ll_f, 'r');
#print (sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f))


# Again gaussian models with full covariance matrices. Now testing a female utterance

ll_m = logpdf_gauss(test_f[1], *train_gauss(train_m)) 
ll_f = logpdf_gauss(test_f[1], *train_gauss(train_f))
# '*' before 'train_gauss' pases both return values (mean and cov) as parameters of 'logpdf_gauss' 
posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f);
plt.figure(); plt.plot(posterior_m, 'b'); plt.plot(1-posterior_m, 'r');
plt.figure(); plt.plot(ll_m, 'b');        plt.plot(ll_f, 'r');
#print (sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f))


# Now run recognition for all male test utterances
# To do the same for females set "test_set=test_f"
score=[]
mean_m, cov_m = train_gauss(train_m)
mean_f, cov_f = train_gauss(train_f)
for tst in test_m:
    ll_m = logpdf_gauss(tst, mean_m, cov_m)
    ll_f = logpdf_gauss(tst, mean_f, cov_f)
    score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))
print(score)

# Run recognition with 1-dimensional LDA projected data
score=[]
mean_m, cov_m = train_gauss(train_m.dot(e))
mean_f, cov_f = train_gauss(train_f.dot(e))
for tst in test_m:
    ll_m = logpdf_gauss(tst.dot(e), mean_m, np.atleast_2d(cov_m))
    ll_f = logpdf_gauss(tst.dot(e), mean_f, np.atleast_2d(cov_f))
    score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))
print(score)


# Train and test with GMM models with diagonal covariance matrices
# Decide for number of gaussian mixture components used for the male model
M_m = 5

# Initialize mean vectors, covariance matrices and weights of mixture componments
# Initialize mean vectors to randomly selected data points from corresponding class
MUs_m  = train_m[randint(1, len(train_m), M_m)]

# Initialize all variance vectors (diagonals of the full covariance matrices) to
# the same variance vector computed using all the data from the given class
COVs_m = [np.var(train_m, axis=0)] * M_m

# Use uniform distribution as initial guess for the weights
Ws_m   = np.ones(M_m) / M_m;


# Initialize parameters of feamele model
M_f = 5
MUs_f  = train_f[randint(1, len(train_f), M_f)]
COVs_f = [np.var(train_f, axis=0)] * M_f
Ws_f   = np.ones(M_f) / M_f;

# Run 30 iterations of EM algorithm to train the two GMMs from males and females
for jj in range(30):
  [Ws_m, MUs_m, COVs_m, TTL_m] = train_gmm(train_m, Ws_m, MUs_m, COVs_m); 
  [Ws_f, MUs_f, COVs_f, TTL_f] = train_gmm(train_f, Ws_f, MUs_f, COVs_f); 
  print('Iteration:', jj, ' Total log-likelihood:', TTL_m, 'for males;', TTL_f, 'for frmales')

# Now run recognition for all male test utterances
# To do the same for females set "test_set=test_f"
score=[]
for tst in test_m:
    ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
    ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
    score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))
print(score)

score=[]
for tst in test_f:
    ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
    ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
    score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))
print(score)
