import imutils
import matplotlib.pyplot as plt
from sklearn import svm
from glob import glob
from skimage.feature import hog
from skimage import data, exposure
from sklearn.externals import joblib
import cv2
from scipy import ndimage
import json






with open('../config.json') as config_file:
    data = json.load(config_file)

orientations =  data['orientations']
pixels_per_cell = data['pixels_per_cell']
cells_per_block = data['cells_per_block']
visualize =  bool(data['visualize'])
transform_sqrt = bool(data['transform_sqrt'])
visualize_test = bool(data['visualize_test'])
flag_print = bool(data['flag_print'])
flag_rotate = bool(data['flag_rotate'])
non_target_train_f = data["non_target_train_f"]
target_train_f = data["target_train_f"]
image_classifier = data["image_classifier"]





def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        f = f.replace("/", "\\")
        print('Processing file: ', f)
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dim = (80, 80)
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        edged = imutils.auto_canny(resized)
        features[f] = edged
    return features


def extract_features(path , datas , labels , flag):
    print("Calculating the descriptors for the " + str(flag)+ " samples and saving them")
    for image in path.keys():
        image_ = path[image]

        fd , hog_image  = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)

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
                    fd, hog_image = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize
                                        )#,transform_sqrt=transform_sqrt
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

        print("Finished hog of ", image.split('\\')[-1])

datas = []
labels = []
non_target_train = png2fea(non_target_train_f)
target_train = png2fea(target_train_f)

extract_features(non_target_train,datas,labels,0)
extract_features(target_train,datas ,labels,1)

print("Training a Linear SVM Classifier")
svc =svm.SVC(kernel='linear', C=5.0,probability=True) # svm.SVC(kernel='linear', C=1.0)
svc.fit(datas,labels)
print("Finished training of Linear SVM Classifier")
joblib.dump(svc, image_classifier)
print("Saving Linear SVM Classifier")



