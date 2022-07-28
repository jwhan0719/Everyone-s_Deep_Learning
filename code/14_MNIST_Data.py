import numpy as np

import tensorflow as tf
import tensorflow.keras.utils as np_utils
from tensorflow.keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('학습셋 이미지 수 : %d 개' % (X_train.shape[0]))
print('테스트셋 이미지 수 : %d 개' % (X_test.shape[0]))

# 가로 28, 세로 28 2차원 배열 -> 784개 1차원 배열로 변환
X_train = X_train.reshape(X_train.shape[0], 784) # reshape(총 샘플수, 1차원 속성의 수)
# 데이터 타입 실수형으로 변환
X_train = X_train.astype('float64')
# 0~255까지의 정수의 데이터를 정규화를 위해 255로 나눔
X_train = X_train / 255

# test 데이터에도 변환 동일 적용
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255


# y 데이터값 확인
print('y 데이터 값 : %d' % (y_train[0]))

# y값 바이너리 변환
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 바이너리 변환된 y 데이터값 확인
print('변환된 y 데이터 값 :', y_train[0])