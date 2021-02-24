import os
import cv2
import numpy as np
from time import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import time

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

def load_images(directory):
    X = [] #image
    Y = [] #label
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (64,64))
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
# (291, 64, 64, 3) (291, 29)
print(X_eval, Y_eval)

'''
 [[0.7372549  0.7647059  0.8235294 ]
   [0.7529412  0.77254903 0.83137256]
   [0.74509805 0.7647059  0.8235294 ]
   ...
   [0.45490196 0.6117647  0.8784314 ]
   [0.47058824 0.6392157  0.9019608 ]
   [0.48235294 0.654902   0.90588236]]]] 

[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
 '''

print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)
print(" Max value of X: ",X.max())
print(" Min value of X: ",X.min())
print(" Shape of X: ",X.shape)

print("\n Max value of Y: ",Y.max())
print(" Min value of Y: ",Y.min())
print(" Shape of Y: ",Y.shape)

# Max value of X:  1.0
#  Min value of X:  0.0
#  Shape of X:  (87000, 64, 64, 3)

#  Max value of Y:  1.0
#  Min value of Y:  0.0
#  Shape of Y:  (87000, 30)
'''
plt.figure(figsize=(24,8))
# A
plt.subplot(2,5,1)
plt.title(Y[0].argmax())
plt.imshow(X[0])
plt.axis("off") # 선 없애는 것
# B
plt.subplot(2,5,2)
plt.title(Y[4000].argmax())
plt.imshow(X[4000])
plt.axis("off")
# C
plt.subplot(2,5,3)
plt.title(Y[7000].argmax())
plt.imshow(X[7000])
plt.axis("off")

plt.suptitle("Example of each sign", fontsize=20)
# plt.show()
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

model = Sequential()

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))
for layer in model.layers:
     layer.trainable = False
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))
for layer in model.layers:
     layer.trainable = False
model.summary()
"""
#model.trainable = True
# 다시 훈련시킨다. 다시 훈련을 15,778,653번 한다는것. 가중치가 틀어진다. 가중치가 틀어지면 최적의 가중치 값을 받은게 소용없어짐.
'''
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 2, 2, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 512)               1049088
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 15,778,653
Trainable params: 15,778,653
Non-trainable params: 0 -> 훈련할수 없는 파라미터
_________________________________________________________________
'''
# model.trainable = False
# 이미지넷에서 얻은 학습된 값을 그대로 사용한다.
# 만약 flatten 이후 dense만 있으면 덴스의 파라미터만 다 더해서 훈련가능한 파라미터.

# for layer in model.layers:
#      layer.trainable = False
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 2, 2, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 512)               1049088
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 15,778,653
Trainable params: 1,063,965
Non-trainable params: 14,714,688
_________________________________________________________________
'''


from keras.optimizers import Adam,RMSprop,Adadelta,Nadam,SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/VGG16_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start = time.time()

op = SGD(lr=0.01)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 64, validation_split=0.2)

model.save('../project/h5/VGG16.hdf5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%') 

print('작업 수행된 시간 : %f 초' % (time.time() - start))

results = model.evaluate(X_eval, Y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])

plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])
plt.show()

# Accuracy for test images: 99.981 %
# Accuracy for evaluation images: 52.414 %
# 작업 수행된 시간 : 851.081062 초
# acc :  0.9996950626373291
# val_acc :  0.9998905062675476
# loss :  0.0011839941143989563
# val_loss :  0.0007794968551024795

''' 
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 62, 62, 64)        1792
_________________________________________________________________
dropout (Dropout)            (None, 62, 62, 64)        0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 29, 29, 64)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 128)       73856
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 12, 128)       0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 128)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 256)         295168
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 512)               524800
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 947,421
Trainable params: 947,421
Non-trainable params: 0
'''

# adam
# Accuracy for test images: 98.41 %
# 작업 수행된 시간 : 650.318738 초
# Accuracy for evaluation images: 30.46 %
# acc :  0.06817381829023361
# val_acc :  0.07365705072879791
# loss :  3.237339496612549
# val_loss :  3.2303311824798584

#adam 128
# Accuracy for test images: 3.448 %
# 작업 수행된 시간 : 4380.264797 초
# Accuracy for evaluation images: 3.448 %
# acc :  0.034600045531988144
# val_acc :  0.034365471452474594
# loss :  3.367187261581421
# val_loss :  3.367766857147217
'''
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 4, 4, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0
_________________________________________________________________
dense (Dense)                (None, 512)               4194816
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 18,924,381
Trainable params: 18,924,381
Non-trainable params: 0
_________________________________________________________________
'''
#sgd 128*128
# Accuracy for test images: 99.989 %
# 작업 수행된 시간 : 11293.462001 초
# Accuracy for evaluation images: 44.713 %
# acc :  1.0
# val_acc :  0.9999101758003235
# loss :  0.0021712803281843662
# val_loss :  0.0027807685546576977

# 기본모델
# Layer (type)                 Output Shape              Param #
# =================================================================
# vgg16 (Functional)           (None, 4, 4, 512)         14714688
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0


# sgd(0.01) - trainable = True
# Accuracy for test images: 99.994 %
# 작업 수행된 시간 : 1072.073220 초
# Accuracy for evaluation images: 50.0 %
# acc :  1.0
# val_acc :  0.9999101758003235
# loss :  3.87013606086839e-05
# val_loss :  0.00016265589511021972

# sgd(0.01) - trainable = False
# Accuracy for test images: 99.655 %
# 작업 수행된 시간 : 1263.428663 초
# Accuracy for evaluation images: 16.207 %
# acc :  0.995195746421814
# val_acc :  0.9965876340866089
# loss :  0.03614778816699982
# val_loss :  0.02695278823375702
"""