# some time later...
import imutils
import matplotlib.pyplot as plt
from prettytable import PrettyTable
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
import pickle


min_wdw_sz = [100, 40]
step_size = [10, 10]
orientations =  8
pixels_per_cell = [16, 16]
cells_per_block = [4, 4]
visualize =  True
transform_sqrt = True
visualize_test =  True
flag_print = False
flag_rotate = True


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        f = f.replace("/", "\\")
        print('Processing file: ', f)
        # features[f] = imread(f,as_gray=False).resize(800,400)
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dim = (80, 80)
        # resize image
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        edged = imutils.auto_canny(resized)
        features[f] = edged
    return features





def evauluate(path,model):
    names  = []
    scores = []
    labels = []
    datas = []
    for image in path.keys():
        print("Image ",image)

        image_ = path[image]

        fd , hog_image  = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        #fd, hog_image = hog(image_, orientations=8, pixels_per_cell=(16, 16),
        #          cells_per_block=(1, 1), visualize=True, multichannel=False)




        #wprint(fd.size)
        #np.append(datas,fd)
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

        datas.append(fd)
        label = model.predict(fd.reshape(1, -1))[0]
        score = model.predict_proba(fd.reshape(1, -1))
        if label == 0 :
            score = str(score[0][0]) +"%"
        else:
            score = str(score[0][1]) + "%"
        names.append(image)
        labels.append(label)
        scores.append(score)
        #print("finishedimage", image)

    t = PrettyTable()
    t.add_column("Names",names)
    t.add_column("Labels",labels)
    t.add_column("Scores",scores)
    print(t)








# load the model from disk
loaded_model = joblib.load("../svm.model")
non_target_dev = png2fea('../data/non_target_dev')
target_dev = png2fea('../data/target_dev')
eval = png2fea('../data/eval')

evauluate(target_dev,loaded_model)
evauluate(non_target_dev,loaded_model)
evauluate(eval,loaded_model)




