# https://www.kaggle.com/gargimaheshwari/asl-recognition-with-deep-learning/log

import numpy as np
np.random.seed(5) 
import tensorflow as tf
tf.random.set_seed(2)  # 다른 컴퓨터에서도 seed 고정 가능
import matplotlib.pyplot as plt
import os
import cv2


train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

def load_images(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (64, 64))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)


import keras

uniq_labels = sorted(os.listdir(train_dir)) # sorted() : 정렬함수
images, labels = load_images(directory = train_dir)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, y_eval = load_images(directory = eval_dir)
 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)
# statify(계층을 짓다) => 계층적 데이터 추출 옵션(분류 모델에서 추천), 여러 층으로 분할 후 각 층별로 랜덤 데이터 추출. 
#                        원래 데이터의 분포와 유사하게 데이터 추출
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)

n = len(uniq_labels)
train_n = len(X_train)
test_n = len(X_test)

print("Total number of symbols: ", n) # 29 (29개 분류. a~z + del + space + nothing)
print("Number of training images: " , train_n) # 78300 (총 87000개의 이미지 중 90% 트레인)
print("Number of testing images: ", test_n) # 8700 (총 87000개의 이미지 중 10% 테스트)

eval_n = len(X_eval)
print("Number of evaluation images: ", eval_n) # 870

# 이미지 확인
def print_images(image_list):
    n = int(len(image_list) / len(uniq_labels))
    cols = 8
    rows = 4
    fig = plt.figure(figsize = (24, 12))

    for i in range(len(uniq_labels)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    # plt.show()

y_train_in = y_train.argsort() # 작은 값부터 순서대로 데이터의 index를 반환해준다. 
y_train = y_train[y_train_in]
X_train = X_train[y_train_in]

# print("Training Images: ")
# print_images(image_list = X_train)

# print("Evaluation images: ")
# print_images(image_list = X_eval)

# a =0, b=1, c=3, d=4 ...
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_eval = to_categorical(y_eval)
y_val = to_categorical(y_val)

print(y_train[0])
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0.]
print(len(y_train[0])) #29

# 데이터 전처리(0~1사이로)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
X_eval = X_eval.astype('float32')/255.0
X_val = X_val.astype('float32')/255.0

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(256, 5, padding = 'same', input_shape = (64, 64, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,5,padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.2))
model.add(Conv2D(128,5, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,5, padding = 'same'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 5, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(29, activation='softmax'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 64, 64, 256)       19456
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 64, 256)       1024
_________________________________________________________________
activation (Activation)      (None, 64, 64, 256)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 256)       1638656
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 256)       1024
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 256)       0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 256)       0
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 256)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       819328
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       409728
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
batch_normalization_3 (Batch (None, 4, 4, 128)         512
_________________________________________________________________
activation_3 (Activation)    (None, 4, 4, 128)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 128)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 64)          204864
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 64)          256
_________________________________________________________________
activation_4 (Activation)    (None, 4, 4, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 128)               131200
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_2 (Dense)              (None, 29)                1885
=================================================================
Total params: 3,236,701
Trainable params: 3,235,037
Non-trainable params: 1,664
_________________________________________________________________
'''
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')

model.compile(optimizer=Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train, y_train, epochs = 5,callbacks=[es,rl], batch_size = 64, validation_data=(X_val,y_val))

model.save('../project/h5/fit.h5')

score = model.evaluate(x = X_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(score[1]*100, 3), '%')                                   
score = model.evaluate(x = X_eval, y = y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')

# Accuracy for test images: 99.713 %
# Accuracy for evaluation images: 39.31 %

#Helper function to plot confusion matrix
def plot_confusion_matrix(y, y_pred):
    y = np.argmax(y, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize = (24, 20))
    ax = plt.subplot()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
    plt.colorbar()
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(uniq_labels))
    plt.xticks(tick_marks, uniq_labels, rotation=45)
    plt.yticks(tick_marks, uniq_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    limit = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment = "center",color = "white" if cm[i, j] > limit else "black")
    plt.show()


from sklearn.metrics import confusion_matrix
import itertools

y_test_pred = model.predict(X_test, batch_size = 64, verbose = 0)
plot_confusion_matrix(y_test, y_test_pred)

y_eval_pred = model.predict(X_eval, batch_size = 64, verbose = 0)
plot_confusion_matrix(y_eval, y_eval_pred)

# Accuracy for test images: 84.741 %
# Accuracy for evaluation images: 29.31 %