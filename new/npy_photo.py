import numpy as np
import tensorflow as tf
import cv2, os 
from tensorflow.keras.utils import to_categorical

train_dir = "A:/study/asl_data/asl_alphabet_train/"
test_dir =  "A:/study/asl_data/asl_alphabet_test/"

def load_images(directory):
    X = [] 
    Y = [] 
    for idx, label in enumerate(asl_labels):
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (128,128))
            X.append(image) 
            Y.append(idx) 
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_classes=29)
    return(X,Y)

asl_labels = sorted(os.listdir(train_dir)) 
X,Y = load_images(train_dir)

if asl_labels == sorted(os.listdir(test_dir)):
    X_TEST, Y_TEST = load_images(test_dir)


# # 트레인 폴더의 파일 리스트와 검증폴더의 파일 리스트가 같다면 트레인 폴더와 같게 x, y를 나눈다.
np.save('A:/study/asl_data/npy/X_TRAIN_128.npy', arr=X)
np.save('A:/study/asl_data/npy/Y_TRAIN_128.npy', arr=Y)
np.save('A:/study/asl_data/npy/X_TEST_128.npy', arr=X_TEST)
np.save('A:/study/asl_data/npy/Y_TEST_128.npy', arr=Y_TEST)


