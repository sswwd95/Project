'''
############################gpu메모리##################################
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
############################################################################
'''
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

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)



# x_train = x_train.reshape(-1,64,64,3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

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
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(29, activation='softmax'))

model.summary()

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')

model.compile(optimizer=Adam(learning_rate=0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 5,callbacks=[es,rl], batch_size = 64, validation_data=(x_val, y_val))

model.save('../project/h5/cnn1.h5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
results = model.evaluate(x = X_eval, y = Y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')
'''
plt.figure(figsize=(24,8))

plt.subplot(1,2,1)
plt.plot(results.history["val_acc"],label="validation_accuracy",c="red",linewidth=4)
plt.plot(results.history["acc"],label="training_accuracy",c="green",linewidth=4)
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(results.history["val_loss"],label="validation_loss",c="red",linewidth=4)
plt.plot(results.history["loss"],label="training_loss",c="green",linewidth=4)
plt.legend()
plt.grid(True)

plt.suptitle("ACC / LOSS",fontsize=18)

plt.show()

'''
'''
def print_images(image_list):
    n=int(len(image_list)/len(Y))
    cols = 8
    rows = 4
    fig = plt.figure(figsize = (24,12))

    for i in range(len(Y)):
        ax = plt.subplot(rows, cols, i+1)
        plt.imshow(image_list[int(n*i)])
        plt.title(Y[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    plt.show()

y_train_in = y_train.argsort()
y_train = y_train[y_train_in]
x_train = x_train[y_train_in]


print("Training Images: ")
print_images(image_list = x_train)

print("Evaluation images: ")
print_images(image_list = X_eval)
'''

# Accuracy for test images: 94.431 %
# Accuracy for evaluation images: 30.92 %


# val 넣었을 때(cnn1)
# Accuracy for test images: 80.695 %
# Accuracy for evaluation images: 27.356 %