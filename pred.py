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

X = np.load('../project/npy/X_train.npy')
Y = np.load('../project/npy/Y_train.npy')
X_eval = np.load('../project/npy/X_eval.npy')
Y_eval = np.load('../project/npy/Y_eval.npy')


model=load_model('../project/h5/cnn1.h5')



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