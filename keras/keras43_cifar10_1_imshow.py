#cifar10 = cnn이 제일 좋음

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train,y_train), (x_test,y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(x_test[0])
print(x_train[0].shape) #(32, 32, 3)
print(np.max(x_train)) #255

plt.imshow(x_train[0])
plt.show()
