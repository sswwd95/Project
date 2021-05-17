
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
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

np.random.seed(42)

X = np.load('A:/study/asl_data/npy/X_TRAIN_64.npy')
Y = np.load('A:/study/asl_data/npy/Y_TRAIN_64.npy')
X_TEST = np.load('A:/study/asl_data/npy/X_TEST_64.npy')
Y_TEST = np.load('A:/study/asl_data/npy/Y_TEST_64.npy')

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
    X, Y, test_size=0.2, random_state=42, stratify = Y)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 29) (17400, 64, 64, 3) (17400, 29)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16

model = Sequential()
model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))
for layer in model.layers:
     layer.trainable = False
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))
model.summary()

'''
model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))
model.summary()
'''

start = datetime.now()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
es=EarlyStopping(patience=8, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(vactor=0.2, patience=4, verbose=1, monitor='val_loss')
filepath = 'A:/study/asl_data/h5/vgg16_64_sgd2_false.h5'
tb = TensorBoard(log_dir='A:/study/asl_data//graph/'+ 'vgg16_64_sgd2_false'+ "/",histogram_freq=0, write_graph=True, write_images=True)
mc = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

op = SGD(lr=0.01)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 1000,callbacks=[es,rl,mc,tb], batch_size = 32,validation_split=0.2)

model.load_weights('A:/study/asl_data/h5/vgg16_64_sgd2_false.h5')

results = model.evaluate(x_test, y_test,batch_size = 32)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
test_results = model.evaluate(X_TEST, Y_TEST,batch_size = 32)
print('Accuracy for evaluation images:', round(test_results[1]*100, 3), '%')
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

# SGD(0.01) : trainable = True
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 51.264 %
# 작업 시간 :  0:18:46.142395

# SGD(0.01) : trainable = False
# Accuracy for test images: 99.931 %
# Accuracy for evaluation images: 15.862 %
# 작업 시간 :  0:19:18.760856
# acc :  0.9990122318267822
# val_acc :  0.9987068772315979
# loss :  0.013599852100014687
# val_loss :  0.009836621582508087

# adam3

# rmsprop3

# adadelta3

# nadam3

# sgd2
