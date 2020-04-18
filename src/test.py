from glob import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import log
import mpl_toolkits.mplot3d
import numpy as np
from numpy import ravel, diag, newaxis, ones, zeros, array, vstack, hstack, dstack, pi, tile
from numpy.linalg import eigh, det, inv, solve, norm
from numpy.random import rand, randn, randint
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.linalg import eigh as eigh_
from matplotlib.image import imread
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import scipy.fftpack

def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        features[f] = imread(f, True).astype(np.float64)
    return features

non_target_dev = png2fea('../data/non_target_dev')
non_target_train = png2fea('../data/non_target_train')
target_dev = png2fea('../data/target_dev')
target_train = png2fea('../data/target_train')



print(non_target_dev)