import os
import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image


df = pd.DataFrame()
labels = []

test = pd.DataFrame()
test_label = []

train_file = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"
for character_file in os.listdir("../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"):
    file = train_file + character_file
    
    if character_file not in ["del","nothing","space"]:
        count = 0
        for img_name in os.listdir(file):
            if count <= 1000:
                img = cv2.imread(file+"/"+img_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_array = gray.reshape(gray.shape[0]*gray.shape[1], 1)

                df[img_name.split(".")[0]] = img_array[:,0]

                labels.append(character_file)

            elif count>1000 and count<=1050:
                img = cv2.imread(file+"/"+img_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_array = gray.reshape(gray.shape[0]*gray.shape[1], 1)
                test[img_name.split(".")[0]] = img_array[:,0]

                test_label.append(character_file[0])

            else:
                break
            count += 1
        
        count=0
    else:
        continue

df = df.T
test = test.T

df.head()

print("len df : ",len(df))
print("len labels :", len(labels))

test.head()

print("len test : ",len(test))
print("len test_label :", len(test_label))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
'''
df = shuffle(df, random_state=42)
labels = shuffle(labels, random_state=42)

test = shuffle(test, random_state=42)
test_label = shuffle(test_label, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_features=500, max_depth=30)

rf.fit(df, labels)
rf_predict = rf.predict(test)

print(classification_report(test_label, rf_predict))

score = round(accuracy_score(test_label, rf_predict), 5)
print("Accurracy: ",score)


############### 저장 ##########################
pickle.dump(rf, open('../project/ml/RF.data', 'wb'))
#dump = 소환, wb = write. 쓰겠다는것
print('저장')

# Accurracy:  0.96231
'''

################ 불러오기 #######################
model = pickle.load(open('../project/ml/RF.data','rb'))
print('불러오기')

test_image=image.load_img("../project/test2/1H0024_test.jpg", target_size=(64,64))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

pred = model.predict(x_predict.reshape())
pred1 = np.argmax(pred,axis=-1)

print(pred)
# pred = []
# for i in pred1:
#     pred.append(uniq_labels[i])
# print('이 손의 뜻은? : ',pred)