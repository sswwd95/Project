import os
import cv2
import numpy as np
from time import time
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import time
start = time.time()

train_dir = '../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_dir = '../procect/asl-alphabet/asl_alphabet_test/asl_alphabet_test'
eval_dir =  "../project/asl-alphabet-test"

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
'''           
plt.figure(figsize=(11, 11))
for i in range (0,29):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    path = train_dir + "/{0}/{0}1.jpg".format(classes[i])
    img = plt.imread(path)
    plt.imshow(img)
    plt.xlabel(classes[i])
'''

def load_data(train_dir):
    images = []
    labels = []
    size = 64,64
    index = -1
    for folder in os.listdir(train_dir):
        index +=1
        for image in os.listdir(train_dir + "/" + folder): # 폴더 불러오기
            img = cv2.imread(train_dir + '/' + folder + '/' + image) # 이미지들 읽기
            img = cv2.resize(img, size) # 이미지 리사이즈
            images.append(img) # 이미지 리스트로 다 모아준다.
            labels.append(index) # 인덱스만 모아서 라벨로 리스트 만들어준다.
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)
    
    print('Loaded', len(x_train),'images for training,','Train data shape =', x_train.shape)
    print('Loaded', len(x_test),'images for testing','Test data shape =', x_test.shape)
    
    return x_train, x_test, y_train, y_test
    

x_train, x_test, y_train, y_test = load_data(train_dir)

from tensorflow.keras.applications import VGG16

model = Sequential()

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))

model.add(Flatten())

model.add(Dense(512, activation='sigmoid'))

model.add(Dense(29, activation='softmax'))

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/VGG16_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 32, validation_split=0.2)

model.save('../project/h5/VGG16.hdf5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%') 

print('작업 수행된 시간 : %f 초' % (time.time() - start))

def load_data(train_dir):
    images = []
    labels = []
    size = 64,64
    index = -1
    for folder in os.listdir(train_dir):
        index +=1
        for image in os.listdir(train_dir + "/" + folder): # 폴더 불러오기
            img = cv2.imread(train_dir + '/' + folder + '/' + image) # 이미지들 읽기
            img = cv2.resize(img, size) # 이미지 리사이즈
            images.append(img) # 이미지 리스트로 다 모아준다.
            labels.append(index) # 인덱스만 모아서 라벨로 리스트 만들어준다.
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = utils.to_categorical(labels)
    return(images, labels)

X_eval, Y_eval = load_data(eval_dir)

results = model.evaluate(X_eval, Y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')

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