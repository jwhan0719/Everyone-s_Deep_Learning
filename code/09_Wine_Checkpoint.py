import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('../data/wine.csv', header=None)

# 데이터프레임 행 무작위 셔플
df = df.sample(frac=1)

X = df.drop(12, axis=1).values
y = df[12].values

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 이진분류 이므로 오차함수는 binary_crossentropy, 최적화 함수는 adam 사용
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = '../model/09/'

# 해당 경로의 폴더가 없을 경우 생성
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)    

# 모델 저장 조건 설정 : 에포크 실행횟수_결과 오차_hdf5
# 모니터할 값 지정 : 학습셋 오차(val_loss)
# 함수 진행사항 출력 : verbose=1
# 앞서 저장한 모델보다 나아졌을 때만 저장 : save_best_only=True
modelpath = '../model/09/{epoch:02d}_{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(X, y, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=checkpointer)
