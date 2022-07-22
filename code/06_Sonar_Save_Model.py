import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('../data/sonar.csv', header=None)
X = df.drop(60,axis=1).values
y = df[60].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=130, batch_size=5)

# 모델 저장
model.save('../model/my_model.h5')

# 테스트를 위해 메모리 내의 모델 삭제
del model

# 모델 호출
model = load_model('../model/my_model.h5')

# 호출한 모델로 테스트 실행
print("\n􀀁 Accuracy:􀀁%.4f" % (model.evaluate(X_test, y_test)[1]))