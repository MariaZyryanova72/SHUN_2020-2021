################
# Бинарная классификация
################

import pandas as pd
from os import path
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

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


################
# Классификация
################

# В Tensorflow есть готовые датасеты для обучения.
# Одним из таких датасетов является датасет с изодражением рукописных цифр

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.colorbar(plt.imshow(train_images[0], cmap='binary_r'))
plt.show()
print(train_labels[0])


from tensorflow.keras.utils import to_categorical

train_labels_categorical = to_categorical(train_labels, 10)
test_labels_categorical = to_categorical(test_labels, 10)
print(train_labels.shape)
print(train_labels_categorical.shape)

print(train_labels_categorical[0])


####################
# softmax функция активации
####################

from tensorflow.keras.layers import Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(train_images, train_labels, batch_size=100, epochs=20)

plt.plot(hist.history['accuracy'])
plt.show()

model.evaluate(test_images, test_labels)
predict = model.predict(test_images[0].reshape(1, 28, 28))
print(predict)

print(np.argmax(predict))
print(test_labels[0])

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(train_images, train_labels_categorical, batch_size=100, epochs=20)

plt.plot(hist.history['accuracy'])
plt.show()

model.evaluate(test_images, test_labels_categorical)
plt.colorbar(plt.imshow(train_images[0], cmap='binary_r'))
plt.show()

train_images = train_images / 255
test_images = test_images / 255
plt.colorbar(plt.imshow(train_images[0], cmap='binary_r'))
plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(train_images, train_labels_categorical, batch_size=100, epochs=20)
model.evaluate(test_images, test_labels_categorical)