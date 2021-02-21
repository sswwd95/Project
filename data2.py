import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import time
np.random.seed(42)

train_dir = "../project/asl-alphabet2/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"

def load_images(directory):
    X = [] #image
    Y = [] #label
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (64,64))
            X.append(image)
            Y.append(idx)
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_classes=29)

    return(X,Y)
'''
print(X, Y)
X =[[[[230   2   4]
    [187   9   9]
    [183  10  14]
    ...
    [190  17  23]
    [188  17  20]
    [212  12  15]]

Y = [ 0  0  0 ... 28 28 28]
'''
   
uniq_labels = sorted(os.listdir(train_dir)) # sorted() -> 정렬함수
X,Y = load_images(directory=train_dir)

print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, Y_eval = load_images(directory=eval_dir)

# 트레인 폴더의 파일 리스트와 검증폴더의 파일 리스트가 같다면 트레인 폴더와 같게 x, y를 나눈다.

print(X_eval.shape, Y_eval.shape) #(870, 64, 64, 3) (870,)
# (291, 64, 64, 3) (291, 29)
print(X_eval, Y_eval)

'''
 [[0.7372549  0.7647059  0.8235294 ]
   [0.7529412  0.77254903 0.83137256]
   [0.74509805 0.7647059  0.8235294 ]
   ...
   [0.45490196 0.6117647  0.8784314 ]
   [0.47058824 0.6392157  0.9019608 ]
   [0.48235294 0.654902   0.90588236]]]] 

[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
 '''

print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)
print(" Max value of X: ",X.max())
print(" Min value of X: ",X.min())
print(" Shape of X: ",X.shape)

print("\n Max value of Y: ",Y.max())
print(" Min value of Y: ",Y.min())
print(" Shape of Y: ",Y.shape)

# Max value of X:  1.0
#  Min value of X:  0.0
#  Shape of X:  (87000, 64, 64, 3)

#  Max value of Y:  1.0
#  Min value of Y:  0.0
#  Shape of Y:  (87000, 30)
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
# plt.show()
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

# x_train = x_train.reshape(-1,64,64,3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, 3, input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(29, activation='softmax'))
model.summary()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/data2_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start = time.time()

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 32, validation_data=(x_val,y_val))

model.save('../project/h5/data2.hdf5')

results = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(results[1]*100, 3), '%')                                   
results = model.evaluate(x = X_eval, y = Y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(results[1]*100, 3), '%')
print('작업 수행된 시간 : %f 초' % (time.time() - start))

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


# Accuracy for test images: 99.983 %
# Accuracy for evaluation images: 46.897 %

# 학습한 데이터만 정답으로 인식한다.-> 과적합

# rmsprop
# Accuracy for test images: 99.966 %
# Accuracy for evaluation images: 47.586 %
# 작업 수행된 시간 : 1000.526890 초

# size0.1
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 41.724 %
# 작업 수행된 시간 : 1334.955268 초

# batch
# Accuracy for test images: 98.759 %
# Accuracy for evaluation images: 24.943 %
# 작업 수행된 시간 : 759.750206 초
# acc :  0.9988789558410645
# val_acc :  0.9876117706298828
# loss :  0.005304587073624134
# val_loss :  0.06209186464548111

# CNN
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 37.931 %
# 작업 수행된 시간 : 1669.383363 초
# acc :  0.9998013377189636
# val_acc :  1.0
# loss :  0.0005857538781128824
# val_loss :  2.4526194465579465e-05

# 아담 0.001
# ccuracy for test images: 99.977 %
# Accuracy for evaluation images: 41.494 %
# 작업 수행된 시간 : 1658.645096 초
# acc :  0.9997304081916809
# val_acc :  1.0
# loss :  0.0013417231384664774
# val_loss :  6.631348242081003e-06

# RMSPROP
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 28.621 %
# 작업 수행된 시간 : 1198.451128 초
# acc :  0.9995175004005432
# val_acc :  1.0
# loss :  0.005928814876824617
# val_loss :  6.172782013891265e-05

#adadelta
# Accuracy for test images: 65.598 %
# Accuracy for evaluation images: 8.506 %
# 작업 수행된 시간 : 1596.654617 초
# acc :  0.5485312938690186
# val_acc :  0.6532567143440247
# loss :  1.4831299781799316
# val_loss :  1.6503159999847412

#nadam
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 38.736 %
# 작업 수행된 시간 : 1727.350156 초
# acc :  0.9998297095298767
# val_acc :  1.0
# loss :  0.0006092878174968064
# val_loss :  5.893501111131627e-06
