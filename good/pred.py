
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
from glob import glob
import cv2
import seaborn as sns
import os
from tensorflow.keras import utils



test_dir = "../project/test"


# def load_data(test_dir):
#     images = []
#     labels = []
#     size = 64,64
#     index = -1
#     for folder in os.listdir(test_dir):
#         index +=1
#         for image in os.listdir(test_dir + "/" + folder): # 폴더 불러오기
#             img = cv2.imread(test_dir + '/' + folder + '/' + image) # 이미지들 읽기
#             img = cv2.resize(img, size) # 이미지 리사이즈
#             images.append(img) # 이미지 리스트로 다 모아준다.
#             labels.append(index) # 인덱스만 모아서 라벨로 리스트 만들어준다.
    
#     images = np.array(images)
#     images = images.astype('float32')/255.0
#     labels = utils.to_categorical(labels)
#     return images, labels

# test_images, test_img_names = load_data()

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
    # names = utils.to_categorical(names)

    return images, names

test_images, test_img_names = load_test_data()

model=load_model('../project/h5/CNN_rms.hdf5')

# make predictions on an image and append it to the list (predictions).
predictions = [model.predict_classes(image.reshape(-1,64,64,3))[0] for image in test_images]

# predictions  = model.predict(test_pred.reshape(-1,64,64,3))
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 1. 0. 0.]]


print(predictions)
# [18, 0, 11, 4] -> SALE

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}

def get_word(predictions):
    predictions_labels = []
    for i in range(len(predictions)):
        for ins in labels_dict:
            if predictions[i] == labels_dict[ins]:
                predictions_labels.append(ins)
                break
    return predictions_labels

result = get_word(predictions)

print(result)
# ['S', 'A', 'L', 'E']

##################### 그림으로 확인 ############################

predfigure = plt.figure(figsize = (10,10))
def plot_image_1(fig, image, label, prediction, predictions_label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label
    plt.title(title)
    return

image_index = 0
row = 1
col = 6
for i in range(1,(row*col-1)):
    plot_image_1(predfigure, test_images[image_index], test_img_names[image_index], predictions[image_index], result[image_index], row, col, i)
    image_index = image_index + 1
plt.show()


