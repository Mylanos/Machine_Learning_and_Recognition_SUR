import matplotlib.pyplot as plt
from sklearn import svm
from glob import glob
from skimage.io import imread
from skimage.feature import hog
from skimage import data, exposure

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import cv2
from skimage.transform import rescale,resize, downscale_local_mean
from scipy import ndimage, misc



min_wdw_sz = [100, 40]
step_size = [10, 10]
orientations =  9
pixels_per_cell = [5, 5]
cells_per_block = [3, 3]
visualize =  False
transform_sqrt = True
visualize_test =  True


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        f = f.replace("/", "\\")
        print('Processing file: ', f)
        features[f] = imread(f, as_gray=True) #rescale(imread(f, as_gray=True), 1.0, anti_aliasing=False)  #.astype(np.float64)
    return features


def extract_features(path , data , labels , flag):
    print("Calculating the descriptors for the " + str(flag)+ " samples and saving them")
    for image in path.keys():
        image_ = path[image]

        # rotate and zoom 
        #fd , hog_image  = hog(image, orientations, pixels_per_cell, cells_per_block, visualize=True)
        fd, hog_image = hog(image_, orientations=8, pixels_per_cell=(4,4), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True)

        np.append(data,fd)
        np.append(labels,flag)
        degree = 0
        if flag_print:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            ax1.axis('off')
            ax1.axis('off')
            ax1.imshow(image_)
            ax1.set_title('Input image')

            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 400))
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()

        while degree < int(360*3.14):
            image_r = ndimage.rotate(image_, degree,reshape=False)
            fd, hog_image = hog(image_r, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2',
                                visualize=True)
            np.append(data,fd)
            np.append(labels,flag)
            if flag_print:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                ax1.axis('off')
                ax1.axis('off')
                ax1.imshow(image_r)
                ax1.set_title('Input image')

                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 400))
                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()


            degree +=5
        print("finishedimage", image)
        #exit()


flag_print = False

data = np.array([])
labels = np.array([])

non_target_train = png2fea('../data/non_target_train')
target_train = png2fea('../data/target_train')

extract_features(target_train,data,labels,1)
extract_features(non_target_train,data,labels,0)


clf = LinearSVC()
print("Training a Linear SVM Classifier")
clf.fit(data, labels)
joblib.dump(clf, "../svm.model")
