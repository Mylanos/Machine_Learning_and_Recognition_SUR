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

        # extract the logo of the car and resize it to a canonical width
        # and height








        #rescale(imread(f, as_gray=True), 1.0, anti_aliasing=False)  #.astype(np.float64)d(f, as_gray=True), 1.0, anti_aliasing=False)  #.astype(np.float64)
    return features


def extract_features(path , datas , labels , flag):
    print("Calculating the descriptors for the " + str(flag)+ " samples and saving them")
    for image in path.keys():
        image_ = path[image]

        # rotate and zoom 
        fd , hog_image  = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        #fd, hog_image = hog(image_, orientations=8, pixels_per_cell=(4,4), cells_per_block=(1, 1), block_norm='L2',
        #                    visualize=True)

        #print(fd.size)
        datas.append(fd)
        labels.append(flag)
        #np.append(datas,fd)
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
        if flag_rotate:
            while degree < int(45*3.14):
                image_r = ndimage.rotate(image_, degree,reshape=False)
                #fd, hog_image = hog(image_r, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2',
                #                    visualize=True)
                fd , hog_image  = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,
                    transform_sqrt=transform_sqrt)
                datas.append(fd)
                labels.append(flag)

                # np.append(datas,fd)
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
                degree +=15
            degree = 0

            while degree > int(-45 * 3.14):
                    image_r = ndimage.rotate(image_, degree, reshape=False)
                    fd, hog_image = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,
                                        transform_sqrt=transform_sqrt)
                    datas.append(fd)
                    labels.append(flag)

                    # np.append(datas,fd)
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

                    degree -=15

        print("finishedimage", image)
    #return np.array(datas) ,labels #datas




datas = []
labels = []

non_target_train = png2fea('../data/non_target_train')
target_train = png2fea('../data/target_train')

#traing_neg ,neg_label =



#traing_pos,pos_label =

extract_features(non_target_train,datas,labels,0)
extract_features(target_train,datas ,labels,1)

#train_set = traing_pos+traing_neg
#train_labels = pos_label + neg_label



#print(train_set , train_labels)
#train_set = np.vstack((traing_pos,traing_neg))
#train_labels = np.concatenate((np.ones(traing_pos.shape[0], ), np.zeros(traing_neg.shape[0], )))


print("Training a Linear SVM Classifier")
svc =svm.SVC(kernel='linear', C=1.0,probability=True) # svm.SVC(kernel='linear', C=1.0)
#clf = LinearSVC()
svc.fit(datas,labels)
print("Finished training of Linear SVM Classifier")
joblib.dump(svc, "../svm.model")
print("Saving Linear SVM Classifier")



