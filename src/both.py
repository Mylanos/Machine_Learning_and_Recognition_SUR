import imutils
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from glob import glob
from skimage.feature import hog
from skimage import exposure
from sklearn.externals import joblib
import cv2
import json
from numba.decorators import jit as optional_jit
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import subprocess


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
both_result = data ["both_result_f"]



eval_extract_flag = bool(data['flag_extract_eval_features'])
result_file = data["audio_classifier_result"]
eval_features_csv = data["eval_features_csv_f"]
audio_classifier = data["audio_classifier"]
eval_folder = data["eval_f"]
target = data["target"]


# deletes content of a given file
def delete_content(file_name):

    with open(file_name, "w"):
        pass


# parses wav files from given directory(directory eval or directory containing non_target/target dirs)
# stores them in a dictionary where key is name of the recording and value is wav file's path
def parse_audio_files(directory):

    sound_files = {}
    print("Loading files in " + directory + " directory..")
    for subdirectory in os.listdir(directory):
        if subdirectory.startswith('target') or subdirectory.startswith('non_target'):
            for filename in os.listdir(f''+directory+'/'+subdirectory):
                if filename.endswith('.wav'):
                    prefix = str(filename[0:7])
                    if prefix in sound_files:
                        sound_files[prefix].append(f''+directory+'/'+subdirectory+"/"+filename)
                    else:
                        sound_files[prefix] = [f''+directory+'/'+subdirectory+"/"+filename]
        if directory == 'eval':
            filename = subdirectory
            if filename.endswith('.wav'):
                prefix = str(filename[0:8])
                if prefix in sound_files:
                    sound_files[prefix].append(f''+directory+'/'+filename)
                else:
                    sound_files[prefix] = [f''+directory+'/'+filename]
    return sound_files


# extracting features from audio files using librosa library
# features are stored in csv 'dataset_file'
def get_features(sound_files, dataset_file):
    delete_content(dataset_file)
    # header defines names of columns each representing audio feature/filename
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' person'
    header = header.split()
    # open csv file
    data_file = open(dataset_file, 'w', newline='')
    with data_file:
        writer = csv.writer(data_file)
        writer.writerow(header)
        for key in sound_files.keys():
            print("Extracting features for " + key + "..\n")
            for file in sound_files[key]:
                songname = f'{file}'
                y, sr = librosa.load(songname, mono=True, duration=10)
                rmse = librosa.feature.rms(y=y)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{file} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {key}'
                writer.writerow(to_append.split())


# divide csv file with features into data and labels
# train_data consists of audio features
# train_labels on training data can be 1(target) or 0 (non_target) on eval data only 0
# person_list list of all filenames
def get_set_of_data(searched_person, dataset_file):
    data = pd.read_csv(dataset_file)
    data.head()
    data = data.drop(['filename'],axis=1)
    person_list = data.iloc[:, -1]
    # transforming recordings to labels(searched person: 1, else: 0)
    y = []
    print("Parsing csv file..")
    for person in person_list:
        if(person.startswith(searched_person)):
            y.append(1)
        else:
            y.append(0)
    train_labels = np.array(y)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    return train_data, train_labels, person_list


# predicts if each record in Data is target or non_target
# Data - records of audio features
# List - list of names each representing one recording
def get_predictions(data, List, table):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    # returns hard decision 0 or 1
    predictions = model.predict_classes(scaled_data, verbose=2, batch_size=len(data))
    # return probabilities for each class
    probabilities = model.predict_proba(scaled_data, verbose=2, batch_size=len(data))

    for i in range(len(data)):
        # pick probability with higher chance
        if probabilities[i, 1] > 0.5:
            probability = probabilities[i, 1]
        else:
            probability = probabilities[i, 0]
        table[List[i]].append(probability)
        table[List[i]].append(predictions[i])


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
    names  = []
    scores = []
    labels = []
    datas = []
    table = {}
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
        #score = str('{0:.0%}'.format (score[0][1]))

        if label == 0 :
            score = score[0][0]
            pass
        else:
            score = score[0][1]

        
        names.append(image.split('/')[-1].split('.')[0] )
        labels.append(label)
        scores.append(score)
        table[image.split('/')[-1].split('.')[0]] =[score,label]
        print("Finished image", image.split('\\')[-1])

    return( table )

    #t = PrettyTable()
    #t.add_column("Names",names)
    #t.add_column("Labels",labels)
    #t.add_column("Scores",scores)
    #print(t)
    #file = open(result,'w')
    #file.write(t.get_string())
    #file.close()
    #print(count)



loaded_model = joblib.load(image_classifier)

eval = png2fea(eval_f)
table = evauluate(eval,loaded_model)


# get evaluating data
eval_files = parse_audio_files(eval_folder)

# if flag is set extract features from files
if eval_extract_flag:
    get_features(eval_files, eval_features_csv)

# testing data(unseen)
eval_data, eval_labels, eval_list = get_set_of_data(target, eval_features_csv)

#load model
model = load_model(audio_classifier)

# Execute predictions on data with loaded model
get_predictions(eval_data, eval_list, table)

#print(table)

file = open(both_result,'w')


for i in sorted(table):
    values = table[i]
    round_percent = 0
    decision = 0 


    image_perc = values[0]
    image_deci = values[1]

    sound_perc = values[2]
    sound_deci = values[3]
    if sound_deci != image_deci:
        treshold = abs(sound_perc-image_perc)
        if( treshold > 25 ):
            if sound_perc > image_perc: 
                decision = sound_deci
            if sound_perc < image_perc:
                decision = image_deci

        else:
            decision = sound_deci

    else:
        decision = sound_deci

    round_percent = (image_perc+sound_perc)/2
    message = str(i) + " " + str(round_percent) + " " + str(decision) +"\n"
    file.write(message)
    #print(i,round_percent,decision)

print(" ")
file.close()







