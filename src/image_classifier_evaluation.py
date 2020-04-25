import imutils
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from glob import glob
from skimage.feature import hog
from skimage import exposure
from sklearn.externals import joblib
import cv2
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
eval_f = data["eval_f"]
result = data['image_classifier_result']



def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        #f = f.replace("/", "\\")
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
    count = 0
    names  = []
    scores = []
    labels = []
    datas = []
    for image in path.keys():
        print("Evaluating image ",image.split('\\')[-1])

        image_ = path[image]
        fd , hog_image  = hog(image_, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)

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
        #label = model.predict(fd.reshape(1, -1))[0]
        label = int((model.predict_proba(fd.reshape(1, -1))[:, 1] >= 0.29).astype(bool))
        score = model.predict_proba(fd.reshape(1, -1))
        score = str('{0:.0%}'.format (score[0][1]))

        if label == 0 :
            #score = str(score[0][0]) + "%"
            pass
        else:
            count+=1

        names.append(image.split('\\')[-1])
        labels.append(label)
        scores.append(score)
        print("Finished image", image.split('\\')[-1])

    t = PrettyTable()
    t.add_column("Names",names)
    t.add_column("Labels",labels)
    t.add_column("Scores",scores)
    print(t)
    file = open(result,'w')
    file.write(t.get_string())
    file.close()
    print(count)





# load the model from disk
loaded_model = joblib.load(image_classifier)
eval = png2fea(eval_f)
evauluate(eval,loaded_model)




