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

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
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
    X, Y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# x_train = x_train.reshape(-1,64,64,3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, 3, input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(256, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='softmax'))
model.summary()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/128_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start = time.time()

op = Adam(lr=0.001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 64, validation_data=(x_val,y_val))

model.save('../project/h5/128.hdf5')

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

# batch
# Accuracy for test images: 98.759 %
# Accuracy for evaluation images: 24.943 %
# 작업 수행된 시간 : 759.750206 초
# acc :  0.9988789558410645
# val_acc :  0.9876117706298828
# loss :  0.005304587073624134
# val_loss :  0.06209186464548111

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

# adam 0.001
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 41.494 %
# 작업 수행된 시간 : 1455.701613 초
# acc :  0.99967360496521
# val_acc :  1.0
# loss :  0.001632230938412249
# val_loss :  7.840417310944758e-06

# adam 0.001
# size =0.3 , dropout=0.3
# Accuracy for test images: 99.985 %
# Accuracy for evaluation images: 46.322 %
# 작업 수행된 시간 : 883.107388 초
# acc :  0.999108612537384
# val_acc :  0.99978107213974
# loss :  0.002451071050018072
# val_loss :  0.0010150223970413208 

# 레이어 늘리기(ppt파일)
# Accuracy for test images: 99.946 %
# Accuracy for evaluation images: 52.299 %
# 작업 수행된 시간 : 1037.638335 초
# acc :  0.9994839429855347
# val_acc :  0.9994526505470276
# loss :  0.0017868331633508205
# val_loss :  0.0022737295366823673

#adam 0.0005
# Accuracy for test images: 99.996 %
# Accuracy for evaluation images: 51.954 %
# 작업 수행된 시간 : 1142.264472 초
# acc :  0.9998592734336853
# val_acc :  0.9998905062675476
# loss :  0.0004584683629218489
# val_loss :  0.0002848122676368803

# 사이즈 64*64, 아담 0.0001
# Accuracy for test images: 99.981 %
# Accuracy for evaluation images: 45.905 %
# 작업 수행된 시간 : 1140.506302 초
# acc :  0.9981468319892883
# val_acc :  0.9998357892036438
# loss :  0.004933896474540234
# val_loss :  0.006093891803175211


# 128*128
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 47.126 %
# 작업 수행된 시간 : 3592.230906 초
# acc :  0.9999640583992004
# val_acc :  0.9998562932014465
# loss :  9.352181950816885e-05
# val_loss :  0.0005202156025916338
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 126, 126, 64)      1792
_________________________________________________________________
dropout (Dropout)            (None, 126, 126, 64)      0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 63, 63, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 61, 61, 64)        36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 61, 61, 64)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 128)       73856
_________________________________________________________________
dropout_2 (Dropout)          (None, 28, 28, 128)       0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 256)       295168
_________________________________________________________________
dropout_3 (Dropout)          (None, 12, 12, 256)       0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0
_________________________________________________________________
dense (Dense)                (None, 512)               4719104
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 29)                14877
=================================================================
=================================================================
Total params: 5,141,725
Trainable params: 5,141,725
Non-trainable params: 0
_______________________________
'''

# stratify 없을경우
# Accuracy for test images: 100.0 %
# Accuracy for evaluation images: 47.931 %
# 작업 수행된 시간 : 885.958147 초
# acc :  0.9996587634086609
# val_acc :  0.9998562932014465
# loss :  0.00113090465310961
# val_loss :  0.0006115080323070288