
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
from tensorflow.keras.utils import to_categorical

test_dir = "A:/study/asl_data/test"

def load_test_data():
    images = []
    names = []
    size = 64,64
    for image in os.listdir(test_dir):
        temp = cv2.imread(test_dir + '/' + image)
        temp = cv2.resize(temp, size)
        images.append(temp)
        names.append(image)
    images = np.array(images).astype('float32')/255.0

    return images, names
    
test_images, test_img_names = load_test_data()
print(test_images, test_img_names)

model=load_model('A:/study/asl_data/h5/adam_4.h5')

predictions = [model.predict_classes(image.reshape(-1,64,64,3))[0] for image in test_images]
# predictions = [model.predict_classes(image.reshape(-1,64,64,3))for image in test_images]
# [array([7], dtype=int64), array([4], dtype=int64), array([11], dtype=int64), array([15], dtype=int64)]

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
        for k in labels_dict:
            if predictions[i] == labels_dict[k]:
                predictions_labels.append(k)
                break
    return predictions_labels

result = get_word(predictions)

print(result)
# ['S', 'A', 'L', 'E']

##################### 그림으로 확인 ############################

# predfigure = plt.figure(figsize = (10,10))
# def plot_image_1(fig, image, label, prediction, predictions_label, row, col, index):
#     fig.add_subplot(row, col, index)
#     plt.axis('off')
#     plt.imshow(image)
#     title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label
#     plt.title(title)
#     return

# image_index = 0
# row = 1
# col = 5
# for i in range(row, col):
#     plot_image_1(predfigure, test_images[image_index], test_img_names[image_index],
#                  predictions[image_index], result[image_index], row, col, i)
#     image_index = image_index + 1
# plt.show()


