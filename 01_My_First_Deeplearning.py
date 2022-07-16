from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# delimiter : 구분자 설정
df = np.loadtxt('../data/ThoraricSurgery.csv', delimiter=",")
X = df[:, 0:17]
y = df[:, 17]

# 딥러닝 구조 결정
model = Sequential() # 딥러닝 구조를 짜고 층을 설정하는 부분

## model.add() : 층 추가 함수, 맨마지막 층을 결과 출력하는 출력층, 나머지는 모두 은닉층
# Dense : 층에 몇개의 노드를 만들 것인지 설정
# input_dim : 입력 데이로부터 몇개의 값이 들어올지 설정
# activation : 다음 층으로 어떻게 값 넘길지 결졍하는 부분
# 30개의 노드, 17개 입력값 -> 데이터에서 17개의 값을 받아 은닉층의 30개 노드로 보냄
model.add(Dense(30, input_dim=17, activation='relu')) # 은닉층 + 입력층 역할
model.add(Dense(1, activation='sigmoid')) # 출력층, 노드수 1개

# loss : 한 번 신경망 실행될 때마다 오차 값 추척하는 함수
# optimizer : 오차 어떻게 줄여 나갈지 정하는 함수
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# epochs(에포크) : 샘플을 몇번 재사용할 것인지 설정
# batch_size : 샘플을 몇개씩 집어넣을 것인지 설정
model.fit(X, y, epochs=10, batch_size=10) # 10번 실행, 샘플 10개씩 

print('\n Accuracy: %.4f' % (model.evaluate(X, y)[1]))