
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os #pip install opencv-python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

np.random.seed(42)

X = np.load('A:/study/asl_data/npy/X_TRAIN3_100.npy')
Y = np.load('A:/study/asl_data/npy/Y_TRAIN3_100.npy')
X_TEST = np.load('A:/study/asl_data/npy/X_TEST3_100.npy')
Y_TEST = np.load('A:/study/asl_data/npy/Y_TEST3_100.npy')

print(X.shape, Y.shape) # (157661, 100, 100, 3) (157661, 29)
print(X_TEST.shape, Y_TEST.shape) # (942, 100, 100, 3) (942, 29) 

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
plt.show()
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 29) (17400, 64, 64, 3) (17400, 29)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet

model = Sequential()

model.add(MobileNet(weights='imagenet', include_top=False, input_shape=(100,100,3)))
for layer in model.layers:
     layer.trainable = False
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))
model.summary()


start = datetime.now()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
es=EarlyStopping(patience=8, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(vactor=0.2, patience=4, verbose=1, monitor='val_loss')
filepath = 'A:/study/asl_data/h5/t3_100_adam3_mobile.h5'
tb = TensorBoard(log_dir='A:/study/asl_data//graph/'+ 't3_100_adam3_mobile' + "/",histogram_freq=0, write_graph=True, write_images=True)
mc = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 1000,callbacks=[es,rl,mc,tb], batch_size = 32,validation_split=0.2)

model.load_weights('A:/study/asl_data/h5/t3_100_adam3_mobile.h5')

results = model.evaluate(x_test, y_test,batch_size = 32)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
test_results = model.evaluate(X_TEST, Y_TEST,batch_size = 32)
print('Accuracy for evaluation images:', round(test_results[1]*100, 3), '%')
end = datetime.now()
time = end - start
print("작업 시간 : " , time)  

# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])
# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])

# plt.plot(acc)
# plt.plot(val_acc)
# plt.plot(loss)
# plt.plot(val_loss)

# plt.title('loss & acc')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')

# plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])
# plt.show()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
mobilenet_1.00_224 (Function (None, 3, 3, 1024)        3228864
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0
_________________________________________________________________
dense (Dense)                (None, 512)               4719104
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
Total params: 7,962,845
Trainable params: 4,733,981
Non-trainable params: 3,228,864
_________________________________________________________________
'''