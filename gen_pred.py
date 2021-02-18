from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten
from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint


train_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

test_gen = ImageDataGenerator(rescale=1./255)

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

batch_size = 64
train_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical'               
)

test_data = test_gen.flow_from_directory(
                eval_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical'
)

classes = list(train_data.class_indices)
print(classes)
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model=load_model('../project/h5/gen.hdf5')

score = model.evaluate_generator(test_data)
print('Accuracy for test images:', round(score[1]*100, 3), '%')
# Accuracy for test images: 29.31 %

# test_image=image.load_img("../project/test/A_test.jpg", target_size=(64,64))
# test_image=image.img_to_array(test_image)
# x_predict=np.expand_dims(test_image, axis=0)

# pred = model.predict(x_predict)


sample = np.array(Image.open("../project/test/A_test.jpg"))
sample_fed = np.expand_dims(sample, 0)
pred= model.predict(sample_fed)
pred = classes[np.argmax(pred)]
plt.imshow(sample)
plt.title("Actual: A, Predicted: {}".format(pred))
plt.axis('off')
plt.show()
