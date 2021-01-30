################
# Бинарная классификация
################

import pandas as pd
from os import path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.utils import plot_model

data = pd.read_csv(path.join('data', 'train.csv'), sep=';')
print(data.head())

y = data['cardio']
data = data[list(data.columns[:-1])]
del data['id']

from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(64, input_dim=11, activation='relu'),
    BatchNormalization(),
    Dense(126, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])

model.summary()

model.fit(data, y, batch_size=100, epochs=20, validation_split=0.3)