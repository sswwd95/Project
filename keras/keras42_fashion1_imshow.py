#fahsion_mnist = cnn이 제일 좋음

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(x_train[0])
print(x_test[0])
print(x_train[0].shape) #(28, 28)
print(np.max(x_train)) #255 

plt.imshow(x_train[0],'gray')
plt.show()
