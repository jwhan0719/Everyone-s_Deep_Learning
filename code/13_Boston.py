import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 공백으로 구분 된 파일 로드방법 : delim_whitespace=True
df = pd.read_csv('../data/housing.csv', delim_whitespace=True, header=None)
X = df.drop(13, axis=1).values
y = df[13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=10)

# 예측 값과 실제 값 비교
# faltten() : 데이터 배열 1차원 변환 함수
y_pred = model.predict(X_test).flatten()
for c in range(10):
    label = y_test[c]
    prediction = y_pred[c]
    print('실제가격: {:.3f}, 예상가격: {:.3f}'.format(label, prediction))