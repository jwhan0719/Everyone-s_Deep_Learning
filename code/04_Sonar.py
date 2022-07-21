import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

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

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=5)

print("\n􀀁Accuracy:􀀁%.4f" % (model.evaluate(X, y)[1]))