
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
import autokeras as ak
np.random.seed(42)

X = np.load('A:/study/asl_data/npy/X_TRAIN2_100.npy')
Y = np.load('A:/study/asl_data/npy/Y_TRAIN2_100.npy')
X_TEST = np.load('A:/study/asl_data/npy/X_TEST2_100.npy')
Y_TEST = np.load('A:/study/asl_data/npy/Y_TEST2_100.npy')

print(X.shape, Y.shape) # (87000, 64, 64, 3) (87000, 29)
print(X_TEST.shape, Y_TEST.shape) # (870, 64, 64, 3) (870, 29)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 29) (17400, 64, 64, 3) (17400, 29)

model = ak.ImageClassifier(
    max_trials=2,
    metrics = 'acc',
    
)

model.fit(x_train, y_train, epochs=10) 

result = model.evaluate(x_test, y_test)


start = datetime.now()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
es=EarlyStopping(patience=8, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(vactor=0.2, patience=4, verbose=1, monitor='val_loss')
filepath = 'A:/study/asl_data/h5/t2_100_adam3.h5'
tb = TensorBoard(log_dir='A:/study/asl_data//graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
mc = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,mc,tb], batch_size = 32,validation_split=0.2)

model.load_weights('A:/study/asl_data/h5/t2_100_adam3.h5')

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

# 64,64
# 544/544 [==============================] - 1s 2ms/step - loss: 2.7974e-04 - accuracy: 0.9999
# Accuracy for test images: 99.989 %
# 28/28 [==============================] - 0s 4ms/step - loss: 7.0340 - accuracy: 0.5069 
# Accuracy for evaluation images: 50.69 %

# 128,128(adam_128) -> (87520,29)
# Accuracy for test images: 99.691 %
# 14/14 [==============================] - 0s 14ms/step - loss: 3.5009 - accuracy: 0.4322
# Accuracy for evaluation images: 43.218 %
# 작업 시간 :  0:12:39.617341
# acc :  0.999875009059906
# val_acc :  0.996715247631073
# loss :  0.00038567616138607264
# val_loss :  0.0173849705606699

# 128,128(adam_128) -> (87000,29)
# 544/544 [==============================] - 3s 5ms/step - loss: 0.0338 - accuracy: 0.9955
# Accuracy for test images: 99.546 %
# 28/28 [==============================] - 0s 7ms/step - loss: 3.9767 - accuracy: 0.3816        
# Accuracy for evaluation images: 38.161 %
# 작업 시간 :  0:07:30.967985
# acc :  0.9992815852165222
# val_acc :  0.9832614660263062
# loss :  0.00233285129070282
# val_loss :  0.06473253667354584

# 위에서 노드 변경
# Accuracy for test images: 99.994 %
# 28/28 [==============================] - 0s 6ms/step - loss: 10.3671 - accuracy: 0.3862
# Accuracy for evaluation images: 38.621 %
# 작업 시간 :  0:14:28.750350
# acc :  0.9999820590019226
# val_acc :  1.0
# loss :  0.00013938565098214895
# val_loss :  8.552351573598571e-06

# X_TRAIN2_64.npy
# (157661, 64, 64, 3) (157661, 29)
# (942, 64, 64, 3) (942, 29)
# Accuracy for test images: 99.045 %
# 30/30 [==============================] - 0s 3ms/step - loss: 4.8022 - accuracy: 0.4246
# Accuracy for evaluation images: 42.463 %
# 작업 시간 :  0:08:31.582543
# acc :  0.9977007508277893
# val_acc :  0.9910410046577454
# loss :  0.007839391939342022
# val_loss :  0.042356472462415695

# X_TRAIN2_100.npy
# Accuracy for test images: 99.182 %
# 30/30 [==============================] - 0s 6ms/step - loss: 4.6523 - accuracy: 0.3758
# Accuracy for evaluation images: 37.58 %
# 작업 시간 :  0:00:22.854229