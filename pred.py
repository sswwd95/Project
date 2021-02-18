
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from PIL import Image


train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
eval_dir =  "../project/asl-alphabet-test"
test_dir = "../project/test"

def load_images(directory):
    X = [] #image
    Y = [] #label
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (64,64))
            # 64x64크기의 이미지
            X.append(image)
            Y.append(idx)
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_classes=29)

    return(X,Y)


uniq_labels = sorted(os.listdir(train_dir)) # sorted() -> 정렬함수
X,Y = load_images(directory=train_dir)

if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, Y_eval = load_images(directory=eval_dir)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


#2. 모델 + 훈련
model=load_model('../project/h5/cnn2_history.hdf5')

#3. 평가
loss, acc=model.evaluate(x_test, y_test, batch_size=100)
print('loss : {:.4f}, acc : {:.2f}'.format(loss, acc*100))
# loss : 0.0000, acc : 100.00

prediction = model.predict(x_test)
print(prediction.shape)
# (8700, 29)


test_image=image.load_img("../project/test/C_test.jpg", target_size=(64,64))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

pred = model.predict(x_predict)
pred1 = np.argmax(pred,axis=-1)

pred = []
for i in pred1:
    pred.append(uniq_labels[i])
print('이 손의 뜻은? : ',pred)

