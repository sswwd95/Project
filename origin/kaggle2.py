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
'''
2. 데이터 로드
데이터는 디렉터리의 여러 폴더에 보관되므로 추출하여 어레이에 저장해야 합니다. 이 작업을 두 번 할 것이기 때문에, 
교육 데이터에 대해, 그리고 평가 데이터에 대해 다시 한 번, 이 작업에 도움이 되는 함수를 작성하는 것이 타당합니다. 
이것은 아래에 주어진 기능으로, 각각의 개별 이미지를 촬영하여 편리한 크기로 크기를 조정한 후 배열에 추가합니다. 
또한 해당 레이블 코드를 레이블 배열에 추가합니다. 레이블 코드는 단순히 각 레이블과 연결된 숫자입니다.
예를 들어 A는 0, B:1, C:2 등으로 이동합니다. 이것은 라벨 인코딩으로 알려져 있으며, 섹션 4에서 조금 더 자세히 다루겠습니다.
'''

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
'''
이제 도우미 기능이 생겼기 때문에 데이터를 추출하는 데 사용할 준비가 되었습니다.
데이터 세트가 크기 때문에 이 작업은 시간이 오래 걸립니다. 
여기서 GPU 서비스를 사용하고 있지만 이러한 기능은 GPU를 활용할 수 없기 때문에 시간에 도움이 되지 않는다.
이와 같은 대규모 데이터 세트를 처리하는 다른 방법은 이미지 및 모델에 대해 생성기를 사용하는 것이다.
생성기는 이 매뉴얼보다 훨씬 빠르지만 거의 무차별적인 방식으로 데이터에 대한 유사한 쉬운 액세스를 허용하지 않습니다.
어떤 방법을 사용할지는 시간과 공간의 제약에 기초하여 이루어진 판단 통화이다. 나는 여기에 그런 의미 있는 제약조건이 없기 때문에, 더 긴 방법을 사용했다.
'''

import keras

uniq_labels = sorted(os.listdir(train_dir)) # sorted() : 정렬함수
images, labels = load_images(directory = train_dir)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, y_eval = load_images(directory = eval_dir)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)
# statify(계층을 짓다) => 계층적 데이터 추출 옵션(분류 모델에서 추천), 여러 층으로 분할 후 각 층별로 랜덤 데이터 추출. 
#                        원래 데이터의 분포와 유사하게 데이터 추출
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
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
    plt.show()

y_train_in = y_train.argsort() # 작은 값부터 순서대로 데이터의 index를 반환해준다. 
y_train = y_train[y_train_in]
X_train = X_train[y_train_in]

print("Training Images: ")
print_images(image_list = X_train)

print("Evaluation images: ")
print_images(image_list = X_eval)

# a =0, b=1, c=3, d=4 ...
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_eval = to_categorical(y_eval)

print(y_train[0])
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0.]
print(len(y_train[0])) #29

# 데이터 전처리(0~1사이로)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
X_eval = X_eval.astype('float32')/255.0

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu', 
                 input_shape = (64, 64, 3)))
model.add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 128 , kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128 , kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 256 , kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(29, activation='softmax'))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 64, 64, 64)        4864
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 64)        102464
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       204928
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       409728
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 128)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 256)         819456
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0
_________________________________________________________________
dense (Dense)                (None, 29)                118813
=================================================================
Total params: 1,660,253
Trainable params: 1,660,253
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train, y_train, epochs = 5, batch_size = 64)

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
