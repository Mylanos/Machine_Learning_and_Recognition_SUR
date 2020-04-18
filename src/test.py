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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import scipy.fftpack
import cv2
import time

from skimage.feature import hog
from skimage import data, exposure








def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        f = f.replace("/", "\\")
        print('Processing file: ', f)
        features[f] = mpimg.imread(f)

    return features


# non_target_dev = png2fea('../data/non_target_dev')
non_target_train = png2fea('../data/non_target_train')
# target_dev = png2fea('../data/target_dev')


target_train = png2fea('../data/target_train')
person_id_all_data = {}

for i in non_target_train.keys():
    name = i.split('//')[-1].split('_')[0]
    if name not in person_id_all_data:
        person_id_all_data.update({name: []})
        person_id_all_data[name].append(non_target_train[i])
    else:
        person_id_all_data[name].append(non_target_train[i])

# print(person_id_all_data)


for person in person_id_all_data.keys():
    for image in person_id_all_data[person]:
        fd, hog_image = hog(image, orientations=8,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1),
                            visualize=True,
                            multichannel=True
                            )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 40))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        plt.show()

    # print(non_target_dev)
