
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

pred = model.predict(x_test)
print(pred.shape)
# (8700, 29)

'''
test_image=image.load_img("../project/test/A_test.jpg", target_size=(64,64))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

pred = model.predict(x_predict)
pred = np.argmax(pred,axis=1)
print('이 손의 뜻은? : ',pred)

if pred==[i]:print('A')
'''
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}

test_dir = "../project/test/"

def load_test_data():
    images = []
    names = []
    size = 64,64
    for image in os.listdir(test_dir):
        temp = cv2.imread(test_dir + '/' + image)
        temp = cv2.resize(temp, size)
        images.append(temp)
        names.append(image)
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names

test_images, test_img_names = load_test_data()

# make predictions on an image and append it to the list (predictions).
predictions = [model.predict_classes(image.reshape(1,64,64,3))[0] for image in test_images]

def get_labels_for_plot(predictions):
    predictions_labels = []
    for i in range(len(predictions)):
        for ins in labels_dict:
            if predictions[i] == labels_dict[ins]:
                predictions_labels.append(ins)
                break
    return predictions_labels

predictions_labels_plot = get_labels_for_plot(predictions)

predfigure = plt.figure(figsize = (13,13))
def plot_image_1(fig, image, label, prediction, predictions_label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label
    plt.title(title)
    return

image_index = 0
row = 5
col = 6
for i in range(1,(row*col-1)):
    plot_image_1(predfigure, test_images[image_index], test_img_names[image_index], predictions[image_index], predictions_labels_plot[image_index], row, col, i)
    image_index = image_index + 1
plt.show()