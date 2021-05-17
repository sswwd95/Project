import numpy as np
import tensorflow as tf
import cv2, os #pip install opencv-python
from tensorflow.keras.utils import to_categorical

train_dir = "A:/study/asl_data/asl_alphabet_train/"
test_dir =  "A:/study/asl_data/asl_alphabet_test/"

def load_images(directory):
    X = [] #image
    Y = [] #label
    for idx, label in enumerate(asl_labels): # idx = 0, 1, 2 ''' , label = A, B, C '''
        for file in os.listdir(directory +'/' + label):
            filepath = directory + '/' + label + '/' + file
            image = cv2.resize(cv2.imread(filepath), (128,128))
            X.append(image) 
            Y.append(idx) 
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_classes=29)
    # print(Y)
    # [[1. 0. 0. ... 0. 0. 0.]
    # [1. 0. 0. ... 0. 0. 0.]
    # [1. 0. 0. ... 0. 0. 0.]
    # ...
    # [0. 0. 0. ... 0. 0. 1.]
    # [0. 0. 0. ... 0. 0. 1.]
    # [0. 0. 0. ... 0. 0. 1.]]

    return(X,Y)

asl_labels = sorted(os.listdir(train_dir)) # sorted() -> 정렬함수
print(asl_labels)
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

X,Y = load_images(train_dir)
print(X,Y)
print(X.shape, Y.shape) #(87000, 64, 64, 3) (87000,)

if asl_labels == sorted(os.listdir(test_dir)):
    X_TEST, Y_TEST = load_images(test_dir)

print(X_TEST.shape, Y_TEST.shape) #


# # 트레인 폴더의 파일 리스트와 검증폴더의 파일 리스트가 같다면 트레인 폴더와 같게 x, y를 나눈다.
np.save('A:/study/asl_data/npy/X_TRAIN3_100.npy', arr=X)
np.save('A:/study/asl_data/npy/Y_TRAIN3_100.npy', arr=Y)
np.save('A:/study/asl_data/npy/X_TEST3_100.npy', arr=X_TEST)
np.save('A:/study/asl_data/npy/Y_TEST3_100.npy', arr=Y_TEST)


# (157661, 128, 128, 3) (157661, 29)
# (942, 128, 128, 3) (942, 29)
# 메모리 터진다

# 파일 수 j와 z빼고 5000개로 조정
# TRAIN2 80,80
# (135176, 80, 80, 3) (135176, 29)
# (942, 80, 80, 3) (942, 29)

# X_TRAIN3_100
# (135176, 100, 100, 3) (135176, 29)
# (942, 100, 100, 3) (942, 29)