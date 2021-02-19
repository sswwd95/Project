from PIL import Image
import numpy as np
np.random.seed(42)
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten
import time
start = time.time()

train_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

eval_gen = ImageDataGenerator(rescale=1./255)


train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

batch_size = 64
train_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical',
                subset='training'                              
)

val_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical',
                subset='validation'
)
eval_data = eval_gen.flow_from_directory(
                eval_dir,
                target_size=(64,64),
                batch_size = batch_size,
                class_mode = 'categorical'
)

classes = list(train_data.class_indices)
print(classes)
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
print(train_data[0])

print(train_data[0][1].shape, val_data[0][0].shape)
# Found 69600 images belonging to 29 classes.
# Found 17400 images belonging to 29 classes.
# Found 870 images belonging to 29 classes.
# (32, 64, 64, 3) (32, 64, 64, 3)
'''
img = load_img("../project/test/1_test.jpg")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야한다.
for batch in train_data.flow_from_directory(x, batch_size=1, save_to_dir='../project/view', save_prefix='tri', save_format='jpg'):
    i += 1
    if i > 30: 
        break
'''
# np.save('../project/npy/gen_train_x.npy', arr=train_gen[0][0])
# np.save('../project/npy/gen_train_y.npy', arr=train_gen[0][1])
# np.save('../project/npy/gen_val_x.npy', arr=val_data[0][0])
# np.save('../project/npy/gen_val_x.npy', arr=val_data[0][1])

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(29, activation='softmax'))
model.summary()

from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/gen_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit_generator(train_data, 
                            steps_per_epoch=69600//64,
                            epochs = 100, 
                            callbacks=[es,rl,cp], 
                            validation_data=val_data,
                            validation_steps=17400//64)

model.save('../project/h5/gen.hdf5')

score = model.evaluate_generator(val_data)
print('Accuracy for test images:', round(score[1]*100, 3), '%')
# Accuracy for test images: 29.31 %                                   
score = model.evaluate_generator(eval_data)
print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')

print('작업 수행된 시간 : %f 초' % (time.time() - start))


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])
plt.show()


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 62, 62, 128)       3584
_________________________________________________________________
dropout (Dropout)            (None, 62, 62, 128)       0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        73792
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        36928
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 32)          18464
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 64)                32832
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_3 (Dense)              (None, 29)                957
=================================================================
Total params: 172,797
Trainable params: 172,797
Non-trainable params: 0
_________________________________________________________________
'''

# optimizer = Adam
# Accuracy for test images: 90.471 %
# Accuracy for evaluation images: 46.667 %
# 작업 수행된 시간 : 6943.168126 초