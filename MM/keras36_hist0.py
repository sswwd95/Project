import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM

# 1. 데이터
a = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape) #(96, 5)

x = dataset[:,0:4]
y = dataset[:,-1]
print(x.shape, y.shape) #(96, 4) (96,)

x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)   # (96, 4, 1)

# 2. 모델
model = load_model("../data/h5/save_keras35.h5")
model.add(Dense(5, name = 'kingkeras1'))
model.add(Dense(1, name = 'kingkeras2'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='acc')
hist = model.fit(x,y, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
print(hist)
print(hist.history.keys())  # dict_keys(['loss', 'val_loss'])

# metrics = 'acc' 넣으면 
# dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])
# epochs 돌면서 나온 loss값들 나옴

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()
