###############
# Регрессия
###############

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt

noise = np.random.randn(20) * 10
x = np.linspace(-5, 5, 20)
y = 10 * x - 7 + noise
plt.scatter(x, y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
print(x_train.shape)
print(x_test.shape)


###############
# Скорость обучения
###############

def plot_prediction(x_test, y_test, model):
    predictions = model.predict(x_test)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_test, y_test, label='test')
    plt.scatter(x_test, predictions, label='predict')
    plt.legend()
    plt.show()


model = Sequential()
model.add(Dense(2, input_dim=1, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

hist = model.fit(x_train, y_train, epochs=500)
plt.plot(hist.history['loss'])
plt.show()
plot_prediction(x_test, y_test, model)

model = Sequential()
model.add(Dense(2, input_shape=(1,), activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['mae'])

hist = model.fit(x_train, y_train, epochs=500)
plt.plot(hist.history['loss'])
plt.show()

plot_prediction(x_test, y_test, model)


###############
# Остановка обучения
###############

class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['loss'] < 90:
            print('\nloss < 90. Останавливаем обучение.')
            self.model.stop_training = True


model = Sequential()
model.add(Dense(2, input_shape=(1,), activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['mae'])

hist = model.fit(x_train, y_train, epochs=500, callbacks=[CallBack()])
plt.plot(hist.history['loss'])
plt.show()
plot_prediction(x_test, y_test, model)

