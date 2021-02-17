import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from keras.optimizers import Adam

X = np.load('../project/npy/X_train.npy')
Y = np.load('../project/npy/Y_train.npy')
X_eval = np.load('../project/npy/X_eval.npy')
Y_eval = np.load('../project/npy/Y_eval.npy')

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


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

# x_train = x_train.reshape(-1,64,64,3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=256,kernel_size=(5,5),activation="relu",padding="same",input_shape=(64,64,3)))
model.add(Conv2D(filters=256,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Conv2D(filters=128,kernel_size=(4,4),activation="relu",padding="same"))
model.add(Conv2D(filters=128,kernel_size=(4,4),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=(2,2),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(40,activation="relu"))

model.add(Dense(29,activation="softmax"))
model.summary()

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')

model.compile(optimizer=Adam(learning_rate=0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist=model.fit(x_train, y_train, epochs = 5,callbacks=[es,rl], batch_size = 64, validation_data=(x_test,y_test))

model.save('../project/h5/cnn2.h5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
results = model.evaluate(x = X_eval, y = Y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')


# Accuracy for test images: 94.431 %
# Accuracy for evaluation images: 30.92 %