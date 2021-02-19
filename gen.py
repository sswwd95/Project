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

print(train_data[0][0].shape, val_data[0][0].shape)
# Found 69600 images belonging to 29 classes.
# Found 17400 images belonging to 29 classes.
# Found 870 images belonging to 29 classes.

# (32, 64, 64, 3) (32, 64, 64, 3)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(75 , (3,3) , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) ,padding = 'same'))
model.add(Conv2D(50 , (3,3), padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) ,  padding = 'same'))
model.add(Conv2D(25 , (3,3) , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/gen_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(train_data, epochs = 5, callbacks=[es,rl,cp], batch_size = 64, validation_data=val_data)

model.save('../project/h5/gen.hdf5')

score = model.evaluate_generator(val_data)
print('Accuracy for test images:', round(score[1]*100, 3), '%')
# Accuracy for test images: 29.31 %


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
