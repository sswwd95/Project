import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import cv2 # OpenCV는 단일 이미지나 동영상의 이미지를 원하는 결과를 분석 및 추출하기 위한 API
import gc #Gabage Collection
# 파이썬은 c또는 c++과 같이 프로그래머가 직접 메모리를 관리하지 않고 레퍼런스카운트와 가비지콜렉션에 의해 관리된다
# gc는 메모리의 모든 객체를 추적한다. 새로운 객체는 1세대에서 시작하고 객체가 살아남으면 두번째 세대로 간다. 
# 파이썬의 가비지 수집기는 총 3게대이며, 객체는 현재 세대의 가비지 수집 프로세스에서 살아남을 때마다 이전 세대로 이동한다. 
# 각 세대마다 임계값 개수의 개체가 있는데 객체 수가 해당 임계값을 초과하면 가비지 콜렉션이 콜렉션 프로세스를 추적한다. 
# 임계값 : Threshold. 만약 0~255사이에서 127의 임계값 지정하고 127보다 작으면 모두 0으로, 127보다 크면 모두 255로 값을 급격하게 변화시킨다. 
# 객체 : 어떠한 속성값과 행동을 가지고 있는 데이터, 파이썬의 모든 것들(숫자, 문자, 함수 등)은 여러 속성과 행동을 가지고 있는 데이터다. 
# 왜 Garbage Collection은 성능에 영향을 주나
# 객체가 많을수록 모든 가비지를 수집하는 데 시간이 오래 걸린다는 것도 분명하다.
# 가비지 컬렉션 주기가 짧다면 응용 프로그램이 중지되는 상항이 증가하고 반대로 주기가 길어진다면 메모리 공간에 가비지가 많이 쌓인다.
from keras import backend as bek
train = pd.read_csv('../dacon7/train.csv')

from sklearn.model_selection import train_test_split

x_train = train.drop(['id','digit', 'letter'], axis=1).values # numpy로 바꾸기
x_train = x_train.reshape(-1,28,28,1)

x_train = np.where((x_train<=20)&(x_train!=0),0.,x_train)
# 최소값, 최대값, 혹은 조건에 해당하는 색인(index) 값을 찾기 : np.argmin(), np.argmax(), np.where()
# 20이하 값 & 0이 아닌 값 => 0으로 바꾸고 아닌것은 그대로 두라는 조건문

# 흑백 이미지 : 대부분 8bit(화소 하나의 색 표현에 8bit 사용), 각 화소의 화소값은 2**8=256개의 값들 중 하나의 값.
# 즉, 0과 255사이의 값들 중 하나의 값이 된다.(0 = 검정색, 255 흰색)
# RGB : 2**8*2**8*2**8 = 2**24 = 16,777,216가지
x_train = x_train/255.
# == x_train = x_train.astype('float32') 실수형으로 바꾸기

y = train['digit']
y_train = np.zeros((len(y),len(y.unique()))) # 총 행의수 , 10(0~9)
# .unique() : Series의 유일한 값을 return하는 함수. (중복되는 값 모두 나오는 것이 아니라 하나만 나옴)
# np.zeros :()안의 공간만큼 0으로 채워준다.  0으로 초기화된 shape 차원의 ndarray 배열 객체를 반환 

for i, digit in enumerate(y):
    y_train[i,digit]=1
# 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 때 사용(목록 표시)
# 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
# i, digit로 쓰면 인덱스와 원소를 각각 다른 변수에 할당(인자 풀기)

# y_train(2048,10) 
# 0  5    => 0 0 0 0 0 1 0 0 0 0 
# 1  0    => 1 0 0 0 0 0 0 0 0 0
# 2  4    => 0 0 0 0 1 0 0 0 0 0


# 300x300의 grayscale 이미지로 리사이즈
train_224 = np.zeros([2048,100,100,3],dtype=np.float32) 

for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    # converted =  변환 , gray색으로 변환
    resized = cv2.resize(converted,(100,100),interpolation=cv2.INTER_CUBIC)
    # 원본이미지, 결과 이미지 크기, 보간법(cv2.INTER_CUBIC, cv2.INTER_LINEAR 이미지 확대할 때 사용/cv2.INTER_AREA는 사이즈 줄일 때 사용)
    # 보간법(interpolation)이란 통계적 혹은 실험적으로 구해진 데이터들(xi)로부터, 
    # 주어진 데이터를 만족하는 근사 함수(f(x))를 구하고,  이 식을 이용하여 주어진 변수에 대한 함수 값을 구하는 일련의 과정을 의미
    del converted
    train_224[i]=resized
    del resized
    bek.clear_session()
    gc.collect()

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
# ReduceLROnPlateau에서 lr는 이전 epoch가 끝날 때 변경되고, LearningRateScheduler에서 lr는 현재 epoch가 시작될 때 변경된다.
# 둘 다 써보기 reduc :         ,learning:
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

datagen = ImageDataGenerator(width_shift_range=(-1,1),
                             height_shift_range=(-1,1),
                             zoom_range=0.15,
                             validation_split=0.2)

valgen = ImageDataGenerator()

def create_model() : 
    effnet = tf.keras.applications.EfficientNetB3(
        include_top=True,
        weights=None,
        input_shape=(100,100,3),
        classes=10,
        classifier_activation='softmax'
    )
    model=Sequential()
    model.add(effnet)
    # effnet? image Classification 타겟의 굉장히 성능이 좋은 Model
    # include_top : 네트워크 상단에 완전 연결 계층을 포함할지 여부. 기본값은 True입니다.
    # weights : None(무작위 초기화), 'imagenet'(ImageNet 사전 학습) 중 하나 또는로드 할 가중치 파일의 경로입니다. 기본값은 'imagenet'입니다.
    # input_tensor : layers.Input()모델의 이미지 입력으로 사용할 선택적 Keras 텐서 (즉,의 출력 ).
    # input_shape : 선택적 모양 튜플, include_topFalse 인 경우에만 지정됩니다 . 정확히 3 개의 입력 채널이 있어야합니다.
    # pooling : 기능 추출을위한 선택적 풀링 모드 include_top입니다 False. 기본값은 없음입니다. - None모델의 출력이 마지막 컨벌루션 레이어의 4D 텐서 출력이됨을 의미합니다. - avg글로벌 평균 풀링이 마지막 컨벌루션 레이어의 출력에 적용되므로 모델의 출력이 2D 텐서가됩니다. - max글로벌 최대 풀링이 적용됨을 의미합니다.
    # classes : 이미지를 분류 할 선택적 클래스 수로, include_topTrue 인 경우에만 지정 되고 weights인수가 지정 되지 않은 경우에만 지정됩니다. 기본값은 1000 (ImageNet 클래스 수)입니다.
    # classifier_activation : A str또는 호출 가능. "상위"레이어에서 사용할 활성화 함수입니다. include_top=True.이 아니면 무시됩니다 . classifier_activation=None"최상위"레이어의 로짓을 반환하도록 설정 합니다. 기본값은 'softmax'입니다.

    model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['accuracy'])
    return model
initial_learningrate = 2e-3

from sklearn.model_selection import RepeatedKFold
# 폴드를 한번만 나누는 것이 아니고 지정한 횟수(n_repeats)만큼 반복해서 나누게 되고 교차 검증 점수도 반복한 만큼 얻을 수 있습니다. 
# 이때 기본적으로 랜덤하게 나누므로 분할기에 자동으로 Shuffle=True 옵션이 적용됩니다. n_repeats의 기본값은 10입니다.
from sklearn.model_selection import KFold
KFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1
results = np.zeros((20480,10))
def lr_decay(epoch):
    #lr_decay 감소
    return initial_learningrate * 0.99 ** epoch
    # 0.002 * 0.99 **epoch ??
test = pd.read_csv('../dacon7/test/csv')

x_test = test.drop(['id', 'letter'],axis = 1).values
x_test = x_test.reshape(-1,28,28,1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = x_test/255.

test_224 = np.zeros([20480,100,100,3],dtype=np.float32)

for i,s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(100,100),interpolation = cv2.INTER_CUBIC)
    del converted
    test_224[i] = resized
    del resized

bek.clear_session()
gc.collect()



