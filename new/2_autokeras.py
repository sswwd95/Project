
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os #pip install opencv-python
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import autokeras as ak

np.random.seed(42)

X = np.load('A:/study/asl_data/npy/X_TRAIN_150.npy')
Y = np.load('A:/study/asl_data/npy/Y_TRAIN_150.npy')
X_TEST = np.load('A:/study/asl_data/npy/X_TEST_150.npy')
Y_TEST = np.load('A:/study/asl_data/npy/Y_TEST_150.npy')

print(X.shape, Y.shape) # (157661, 100, 100, 3) (157661, 29)
print(X_TEST.shape, Y_TEST.shape) # (942, 100, 100, 3) (942, 29) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (126128, 100, 100, 3) (126128, 29) (31533, 100, 100, 3) (31533, 29)

model = ak.ImageClassifier(
    max_trials=1,
    metrics = 'acc',
    loss = 'categorical_crossentropy',
    directory='A:\\study\\asl_data\\ak\\',
    num_classes=29,
    seed=42  
)


start = datetime.now()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
es=EarlyStopping(patience=8, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(vactor=0.2, patience=4, verbose=1, monitor='val_loss')
filepath = 'A:/study/asl_data/h5/t2_100_ak.h5'
tb = TensorBoard(log_dir='A:/study/asl_data//graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
mc = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=10,callbacks=[es,rl,mc,tb]) 

model.load_weights('A:/study/asl_data/h5/t2_100_ak.h5')

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

model = load_model('A:/study/asl_data/h5/ak_model.h5')
model.summary()
