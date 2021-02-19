
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


test_image=glob('../project/test/*.jpg')
print(len(test_image))

test_pred =[]
for i in test_image:
    img = image.load_img(i, target_size=(64,64))
    img = image.img_to_array(img)
    img1 = np.expand_dims(img, axis=0)
    test_pred.append(img1)
    # test_pred = test_pred.astype('float32')/255.

test_pred=np.array(test_pred)
test_pred = test_pred.astype('float32')/255.0

model=load_model('../project/h5/cnn2_history.hdf5')

pred = model.predict_classes(test_pred.reshape(-1,64,64,3))

# pred = np.argmax(pred,axis=-1)
print(pred)
# [11 14 21 26]

# pred = model.predict(test_pred.reshape(-1,64,64,3))
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
#   0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 1. 0. 0.]]

test_pred2 = []
for i in range(len(pred)):
    test_pred2.append(pred[i])

print(test_pred2)
# [11, 14, 21, 26]

categories = ["A","B", "C", "D", "E", "F","G", "H","I", "J","K", "L", "M",
              "N","O","P","Q","R", "S", "T","U","V","W", "X", "Y", "Z","DEL","NOTIHG","SPACE"]

result= []
for i in test_pred2 :
    str_result = str(categories[i])
    print(str_result)
    result.append(str_result)

print(result)


# 값이 다르게 나온다.
