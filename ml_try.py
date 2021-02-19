import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
# np.random.seed(5) 
import tensorflow as tf
tf.random.set_seed(2)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
test_dir =  "../project/asl-alphabet-test"

def load_data():
    images = []
    labels = []
    size = 64,64
    print("LOADING DATA FROM : ",end = "")
    for folder_index, folder in enumerate(os.listdir(train_dir)):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(folder_index)
        
    images = np.array(images)
    images = images.astype('float32')/255.0
    
    labels = keras.utils.to_categorical(labels)   #one-hot encoding
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.2)

    return X_train, X_test, Y_train, Y_test

x_train, x_test, y_train, y_test = load_data()

