#!/usr/bin/env python
# coding: utf-8

from numba.decorators import jit as optional_jit
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.preprocessing import StandardScaler
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import json

with open('../config.json') as config_file:
    data = json.load(config_file)

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


# get evaluating data
eval_files = parse_audio_files(eval_folder)


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


if eval_extract_flag:
    get_features(eval_files, eval_features_csv)

# testing data(unseen)
Eval_data, Eval_labels, Eval_list = get_set_of_data(target, eval_features_csv)


# predicts if each record in Data is target or non_target
# Data - records of audio features
# List - list of names each representing one recording
def get_predictions(data, List):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    # returns hard decision 0 or 1
    print("Predicting...")
    predictions = model.predict_classes(scaled_data, verbose=2, batch_size=len(data))
    # return probabilities for each class
    probabilities = model.predict_proba(scaled_data, verbose=2, batch_size=len(data))

    results = []
    for i in range(len(data)):
        # pick probability with higher chance
        if probabilities[i, 1] > 0.5:
            probability = probabilities[i, 1]
        else:
            probability = probabilities[i, 0]
        results.append(str(List[i]) + " " + str(probability) + " " + str(predictions[i]) + "\n")

    with open(result_file, "w") as results_file:
        # sort according to eval order
        for i in sorted(results):
            results_file.write(i)
    print("Done!")


# load trained model
model = load_model(audio_classifier)

# Execute predictions on data with loaded model
get_predictions(Eval_data, Eval_list)


