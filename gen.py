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
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

eval_gen = ImageDataGenerator(rescale=1./255)


train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"
# eval_dir =  "../project/ASL"


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
from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(29, activation='softmax'))
model.summary()

from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/gen2_eval_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit_generator(train_data, 
                            steps_per_epoch=69600//64,
                            epochs = 100, 
                            callbacks=[es,rl,cp], 
                            validation_data=val_data,
                            validation_steps=10)

model.save('../project/h5/gen2_eval.hdf5')

score = model.evaluate_generator(val_data)
print('Accuracy for test images:', round(score[1]*100, 3), '%')
score = model.evaluate_generator(eval_data)
print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')

print('작업 수행된 시간 : %f 초' % (time.time() - start))


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc)
print('loss : ', loss)
print('val_loss : ', val_loss)


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
'''______________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 62, 62, 128)       3584
_________________________________________________________________
batch_normalization (BatchNo (None, 62, 62, 128)       512
_________________________________________________________________
dropout (Dropout)            (None, 62, 62, 128)       0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        73792
_________________________________________________________________
batch_normalization_1 (Batch (None, 29, 29, 64)        256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        36928
_________________________________________________________________
batch_normalization_2 (Batch (None, 12, 12, 64)        256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 32)          18464
_________________________________________________________________
batch_normalization_3 (Batch (None, 4, 4, 32)          128
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
Total params: 173,949
Trainable params: 173,373
Non-trainable params: 576
_________________________________________________________________
'''

# batchnormalization
# Accuracy for test images: 90.023 %
# Accuracy for evaluation images: 29.195 %
# 작업 수행된 시간 : 6767.633879 초
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 64, 64, 64)        3136
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 64, 64)        256
_________________________________________________________________
activation (Activation)      (None, 64, 64, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 64, 64, 64)        0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 22, 22, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 22, 22, 64)        65600
_________________________________________________________________
batch_normalization_1 (Batch (None, 22, 22, 64)        256
_________________________________________________________________
activation_1 (Activation)    (None, 22, 22, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 22, 22, 64)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 128)         131200
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 128)         512
_________________________________________________________________
activation_2 (Activation)    (None, 8, 8, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 128)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 128)         262272
_________________________________________________________________
batch_normalization_3 (Batch (None, 3, 3, 128)         512
_________________________________________________________________
activation_3 (Activation)    (None, 3, 3, 128)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 128)         0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 1, 128)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 256)         524544
_________________________________________________________________
batch_normalization_4 (Batch (None, 1, 1, 256)         1024
_________________________________________________________________
activation_4 (Activation)    (None, 1, 1, 256)         0
_________________________________________________________________
dropout_4 (Dropout)          (None, 1, 1, 256)         0
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 256)               0
_________________________________________________________________
dense (Dense)                (None, 512)               131584
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 1,135,773
Trainable params: 1,134,493
Non-trainable params: 1,280
_________________________________________________________________
'''
# barchnormalization(순서 바꾸기) + RMSprop