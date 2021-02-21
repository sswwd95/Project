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
    X = np.array(X)
    Y = np.array(Y)
    return(X,Y)
   
uniq_labels = sorted(os.listdir(train_dir)) 
X,Y = load_images(directory=train_dir)

print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, Y_eval = load_images(directory=eval_dir)

print(X_eval.shape, Y_eval.shape) #(870, 64, 64, 3) (870,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
    

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (69600, 64, 64, 3) (69600, 30) (17400, 64, 64, 3) (17400, 30)

# 이미지 확인
def print_images(image_list):
    n = int(len(image_list) / len(uniq_labels))
    cols = 8
    rows = 4
    fig = plt.figure(figsize = (24, 12))

    for i in range(len(uniq_labels)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    plt.show()

y_train_in = y_train.argsort()
y_train = y_train[y_train_in]
x_train = x_train[y_train_in]

# print_images(image_list = x_train)

# print_images(image_list = X_eval)

# a =0, b=1, c=3, d=4 ...
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(y_train[0])
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0.]
print(len(y_train[0])) #29

# 데이터 전처리(0~1사이로)
x = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
x_val = x_val.astype('float32')/255.0


from keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,BatchNormalization,Activation
from keras.models import Sequential
from tensorflow.keras.applications import VGG16

model = Sequential()

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))

model.add(Flatten())

model.add(Dense(512, activation='sigmoid'))

model.add(Dense(29, activation='softmax'))

model.summary()

from keras.optimizers import Adam,RMSprop,Adadelta,Nadam,SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss',restore_best_weights = True)
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')
filepath = '../project/modelcp/VGG16_2_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start = time.time()

op = SGD(lr=0.0001)
model.compile(optimizer=op, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(x_train, y_train, epochs = 100,callbacks=[es,rl,cp], batch_size = 32, validation_data=(x_val,y_val))

model.save('../project/h5/VGG16_2.hdf5')

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


