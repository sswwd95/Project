import os
import cv2
import numpy as np
from time import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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
            image = cv2.resize(cv2.imread(filepath), (200,200))
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
    X, Y, test_size=0.3, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42
)


from tensorflow.keras.applications import VGG16

model = Sequential()

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3)))

model.add(Flatten())

model.add(Dense(512, activation='sigmoid'))

model.add(Dense(29, activation='softmax'))

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam,SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/VGG16_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start = time.time()

op = SGD(lr=0.0001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 32, validation_split=0.2)

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

# Accuracy for test images: 98.167 %
# 작업 수행된 시간 : 1338.359044 초
# Accuracy for evaluation images: 32.069 %

'''
poch 00001: val_loss improved from inf to 0.12225, saving model to ../project/modelcp\VGG16_0.122.hdf5
1740/1740 [==============================] - 58s 33ms/step - loss: 0.9329 - accuracy: 0.7028 - val_loss: 0.1223 - val_accuracy: 0.9620
Epoch 2/100
1739/1740 [============================>.] - ETA: 0s - loss: 0.1184 - accuracy: 0.9642 
Epoch 00002: val_loss improved from 0.12225 to 0.06091, saving model to ../project/modelcp\VGG16_0.061.hdf5
1740/1740 [==============================] - 58s 34ms/step - loss: 0.1183 - accuracy: 0.9642 - val_loss: 0.0609 - val_accuracy: 0.9818
Epoch 3/100
1739/1740 [============================>.] - ETA: 0s - loss: 0.0884 - accuracy: 0.9750 
Epoch 00003: val_loss did not improve from 0.06091
1740/1740 [==============================] - 58s 33ms/step - loss: 0.0891 - accuracy: 0.9748 - val_loss: 0.9764 - val_accuracy: 0.7271
Epoch 4/100
1739/1740 [============================>.] - ETA: 0s - loss: 3.3986 - accuracy: 0.0516 
Epoch 00004: val_loss did not improve from 0.06091
1740/1740 [==============================] - 58s 33ms/step - loss: 3.3986 - accuracy: 0.0517 - val_loss: 3.3747 - val_accuracy: 0.0576
Epoch 5/100
1739/1740 [============================>.] - ETA: 0s - loss: 3.3413 - accuracy: 0.0581    
Epoch 00005: val_loss did not improve from 0.06091
1740/1740 [==============================] - 58s 34ms/step - loss: 3.3413 - accuracy: 0.0580 - val_loss: 3.3295 - val_accuracy: 0.0637
Epoch 6/100
1739/1740 [============================>.] - ETA: 0s - loss: 3.3177 - accuracy: 0.0642 
Epoch 00006: val_loss did not improve from 0.06091
1740/1740 [==============================] - 58s 34ms/step - loss: 3.3177 - accuracy: 0.0642 - val_loss: 3.3150 - val_accuracy: 0.0652
Epoch 7/100
1739/1740 [============================>.] - ETA: 0s - loss: 3.3077 - accuracy: 0.0653 
'''


