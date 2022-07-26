import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('../data/wine.csv', header=None)

# 전체 샘플 중 15%만 추출
df = df.sample(frac=0.15)

X = df.drop(12, axis=1).values
y = df[12].values

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 학습 자동 중단 설정
# 모니터할 값 지정 : 학습셋 오차(val_loss)
# 성능 향상이 되지 않더라도 학습할 최대 횟수 : 10회
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

model.fit(X, y, validation_split=0.2, epochs=2000, batch_size=500, callbacks=[early_stopping_callback])

print('\n Accuracy: %.4f' % (model.evaluate(X, y)[1]))