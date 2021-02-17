import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import PIL.Image as pilimg
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

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
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_classes=29)

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
print(X_eval, Y_eval)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


model=load_model('../project/h5/cnn2.hdf5')



loss, acc = model.evaluate(X_eval, Y_eval)
print('loss, acc : ', loss, acc)
# loss, acc :  0.30173632502555847 0.90625
'''
image = pilimg.open('../data/image/me/star3.jpg')
pix = image.resize((128,128))
pix = np.array(pix)
test = pix.reshape(1,128,128,3)/255.

pred_answer = [0] # 여자
pred_no_answer = [1] # 남자

pred = model.predict(test)
print('pred : ',pred)

print('여자일 확률은 ', (1-pred)*100, '%')
print('남자일 확률은 ', pred*100, '%')

if pred >0.5:
    print('당신은 남자입니다!')
else:
    print('당신은 여자입니다!')

# 나
# pred :  [[0.00595943]]
# 여자일 확률은  [[99.40405]] %
# 남자일 확률은  [[0.5959431]] %
# 당신은 여자입니다!

# 영리
# pred :  [[0.00538115]]
# 여자일 확률은  [[99.46188]] %
# 남자일 확률은  [[0.53811514]] %
# 당신은 여자입니다!

# 마동석
# pred :  [[0.99944156]]
# 여자일 확률은  [[0.05584359]] %
# 남자일 확률은  [[99.94415]] %
# 당신은 남자입니다!
'''