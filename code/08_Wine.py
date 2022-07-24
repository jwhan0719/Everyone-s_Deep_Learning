import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model.fit(X, y, epochs=200, batch_size=200)

print('\nAccuary %.4f' % (model.evaluate(X, y)[1]))