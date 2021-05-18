import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from datetime import datetime

from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

np.random.seed(42)

X = np.load('A:/study/asl_data/npy/X_TRAIN_64.npy')
Y = np.load('A:/study/asl_data/npy/Y_TRAIN_64.npy')
X_TEST = np.load('A:/study/asl_data/npy/X_TEST_64.npy')
Y_TEST = np.load('A:/study/asl_data/npy/Y_TEST_64.npy')

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
    X, Y, test_size=0.2, random_state=42,stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# x_train = x_train.reshape(-1,64,64,3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, 3, input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(256, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))
model.summary()
start = datetime.now()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint, TensorBoard

es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = 'A:/study/asl_data/h5/CNN_64_SGD2.h5'
mc = ModelCheckpoint(filepath=filepath, monitor='val_loss', 
                    verbose=1, save_best_only=True, mode='auto')
op = SGD(lr=0.01)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 1000,callbacks=[es,rl,mc], 
                  batch_size = 32, validation_data=(x_val,y_val))

model.save('A:/study/asl_data/h5/CNN_64_SGD2.h5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
results = model.evaluate(x = X_TEST, y = Y_TEST, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  
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

# 128,128
# Epoch 00083: val_loss did not improve from 0.00087
# Epoch 00083: early stopping
# Accuracy for test images: 99.989 %
# Accuracy for evaluation images: 47.931 %
# 작업 시간 :  1:50:13.548681
#==========================================================
# 64,64 adam3
# Accuracy for test images: 99.971 %
# Accuracy for evaluation images: 54.713 %
# 작업 시간 :  0:14:48.028487
# acc :  0.9996767044067383
# val_acc :  0.9997126460075378
# loss :  0.002292361343279481
# val_loss :  0.0012665786780416965


# rmsprop3
# Accuracy for test images: 99.989 %
# Accuracy for evaluation images: 54.713 %
# 작업 시간 :  0:20:21.860312
# acc :  0.9992636442184448
# val_acc :  0.9998562932014465
# loss :  0.01449055690318346
# val_loss :  0.00051312759751454

# adadelta3
# Accuracy for test images: 96.511 %
# Accuracy for evaluation images: 21.494 %
# 작업 시간 :  1:48:41.042930
# acc :  0.9201867580413818
# val_acc :  0.965732753276825
# loss :  0.23254600167274475
# val_loss :  0.5545966029167175

# nadam3
# Accuracy for test images: 99.989 %
# Accuracy for evaluation images: 57.241 %
# 작업 시간 :  0:33:14.115242
# acc :  0.9997305870056152
# val_acc :  0.9998562932014465
# loss :  0.0015300079248845577
# val_loss :  0.0004186548467259854

# sgd2
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 41.379 %
# 작업 시간 :  0:35:19.678746
# acc :  0.9998024702072144
# val_acc :  0.9999281764030457
# loss :  0.0008428720757365227
# val_loss :  0.005195867735892534