import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
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

# 10개의 파일로 분리
n_fold = 5
kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# accuracy 리스트 생성
accuracy = []

for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(X[train], y[train], epochs=50, batch_size=5)

    acc = "%.4f" % (model.evaluate(X[test], y[test])[1])
    accuracy.append(acc)


print("\n %.f fold accuracy:" % n_fold, accuracy)    