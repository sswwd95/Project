import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

def load_images(directory):
    X = [] #image
    Y = [] #label
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (64,64))
            # 64x64크기의 이미지
            X.append(image)
            Y.append(idx)
    X = np.array(X)
    Y = np.array(Y)
    return(X,Y)
'''
print(X, Y)
X =[[[[230   2   4]
    [187   9   9]
    [183  10  14]
    ...
    [190  17  23]
    [188  17  20]
    [212  12  15]]

Y = [ 0  0  0 ... 28 28 28]
'''

uniq_labels = sorted(os.listdir(train_dir)) # sorted() -> 정렬함수
X,Y = load_images(directory=train_dir)

print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, Y_eval = load_images(directory=eval_dir)

# 트레인 폴더의 파일 리스트와 검증폴더의 파일 리스트가 같다면 트레인 폴더와 같게 x, y를 나눈다.

print(X_eval.shape, Y_eval.shape) #(870, 64, 64, 3) (870,)
'''
np.save('../project/npy/X_train.npy',X)
np.save('../project/npy/Y_train.npy',Y)
np.save('../project/npy/X_eval.npy',X_eval)
np.save('../project/npy/Y_eval.npy',Y_eval)
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y)

########## val ? ##############################
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2
)
# x_train = X[:69600]
# y_train = Y[:69600]
# x_test = X[:17400]
# y_test = Y[:17400]

# x_val = x_train[:13920]
# y_val = y_train[:13920]
##############################################

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600,) (17400, 64, 64, 3) (17400,) val 분리 x
# (55680, 64, 64, 3) (55680,) (17400, 64, 64, 3) (17400,)
# print(x_val.shape, y_val.shape) #(13920, 64, 64, 3) (13920,)

###################### 이미지 확인 ######################
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
x_train = x_train[y_train_in]

print("Training Images: ")
print_images(image_list = x_train)
# print_images(image_list = x_val)

print("Evaluation images: ")
print_images(image_list = X_eval)
'''
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=30) #num_classes = total number of classes
y_test = to_categorical(y_test, num_classes=30)

# 추가 하위 샘플링을 허용하기 위해 데이터 섞기
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=42)
x_test, y_test = shuffle(x_test, y_test, random_state=42)
x_train = x_train[:]

'''

