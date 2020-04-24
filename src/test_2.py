
from sklearn import svm
from glob import glob
#from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import cv2
from skimage.transform import rescale,resize, downscale_local_mean

threshold = 0.3


import numpy as np



def overlapping_area(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    '''
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    '''
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
            reverse=True)
    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections







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




data = []
labels = []
dev_data = []
results = []


non_target_dev = png2fea('../data/non_target_dev')
non_target_train = png2fea('../data/non_target_train')
target_train = png2fea('../data/target_train')
target_dev = png2fea('../data/target_dev')


def extract_features(path , data , labels , flag):
    print("Calculating the descriptors for the " + str(flag)+ " samples and saving them")
    for image in path.keys():
        image = path[image]
        #print(image)
        fd = hog(image, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        data.append(fd)
        labels.append(flag)

#def get_result( )

right_result = []

extract_features(target_train,data,labels,1)
extract_features(non_target_train,data,labels,0)


extract_features(target_dev,dev_data,right_result,1)
extract_features(non_target_dev,dev_data,right_result,0)


clf = LinearSVC()
print("Training a Linear SVM Classifier")
clf.fit(data, labels)
right_result_count = 0
count = 0
for i in dev_data:
    try:
        result = clf.predict(i.reshape(1, -1))
    except ValueError:
        result = [-1]
    print(result, right_result[count] )
    if ( result[0] == right_result[count] ):
        right_result_count +=1
    count +=1

print("Right result count is ", right_result_count)

exit()



"""
#plt.plot(clf)
#plt.show()







def sliding_window(image, window_size, step_size):
"""
    """
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    """
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
"""


im =  rescale(imread("../data/target_dev/m429_03_p03_i0_0.png", as_gray=False), 2.0, anti_aliasing=False)
clf = joblib.load("../svm.model")
detections = []
scale =0
downscale = 1.25
visualize_det = False
min_wdw_sz = [100, 40]

from skimage.transform import pyramid_gaussian
for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            print('error')
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            if visualize_test==False:
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize_test, transform_sqrt=transform_sqrt)
            elif visualize_test==True:
                fd ,hog_image= hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize_test, transform_sqrt=transform_sqrt)
                cv2.imshow('hog_image',hog_image)
                cv2.waitKey(0)
            pred = clf.predict(fd.reshape(1,-1))
            if pred == 1:
                print ("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd)))
                detections.append((x, y, clf.decision_function(fd),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
            # If visualize is set to true, display the working
            # of the sliding window
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _, _  in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                        im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(30)
        # Move the the next scale
        scale+=1


clone = im.copy()
for (x_tl, y_tl, _, w, h) in detections:
    # Draw the detections
    cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
cv2.imshow("Raw Detections before NMS", im)
cv2.waitKey()

# Perform Non Maxima Suppression
detections = nms(detections, threshold)

# Display the results after performing NMS
for (x_tl, y_tl, _, w, h) in detections:
    # Draw the detections
    cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=10)
cv2.imshow("Final Detections after applying NMS", clone)
cv2.waitKey()


#clf = svm.SVC()
#clf.fit(X, y)
#print(clf.predict( [[2., 2.]] ))
"""