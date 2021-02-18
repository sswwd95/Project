from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten

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
test_dir = "../project/asl-alphabet/asl_alphabet_test/asl_alphabet_test"

eval_dir =  "../project/asl-alphabet-test"

batch_size = 32
train_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical'               
)

test_data = test_gen.flow_from_directory(
                test_dir,
                target_size = (64,64),
                batch_size = batch_size,
                class_mode = 'categorical'
)



classes = list(train_data.class_indices)
print(classes)
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

print(train_data[0][0].shape)#, val_data[0][0].shape)
# Found 69600 images belonging to 29 classes.
# Found 17400 images belonging to 29 classes.
# (32, 64, 64, 3) (32, 64, 64, 3)
'''
from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(64, activation='relu'))
model.add(Dense(29, activation='softmax'))
model.summary()

from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/gen_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(train_data, epochs = 50, callbacks=[es,rl,cp], batch_size = 16, validation_data=val_data)

model.save('../project/h5/gen.hdf5')

score = model.evaluate_generator(val_data)
print('Accuracy for test images:', round(score[1]*100, 3), '%')

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['acc', 'val_acc']].plot();

sample = np.array(Image.open("../project/test/A_test.jpg"))
sample_fed = np.expand_dims(sample, 0)
pred= model.predict(sample_fed)
pred = classes[np.argmax(pred)]
plt.imshow(sample)
plt.title("Actual: A, Predicted: {}".format(pred))
plt.axis('off')
plt.show()
'''