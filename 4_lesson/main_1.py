# Библиотека SKLearn - библиотека для решения задач машинного обучения.
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error
# Нас интересует средняя квадратичная(MSE) и средняя абсолютная ошибка(MAE).
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics

sklearn.metrics.__dir__()

import matplotlib.pyplot as plt

# linear_model и neighbors


pd.set_option('display.max_rows', 100, 'display.max_columns', 200)
# строчка нужна, чтобы датасеты с множеством колонок отображались полностью


#########################
# Линейная регрессия
#########################

# Линейная регрессия (англ. Linear regression) — используемая в статистике регрессионная модель
# зависимости одной (объясняемой, зависимой) переменной. от другой или нескольких других переменных
# (факторов, регрессоров, независимых переменных). с линейной функцией зависимости.

noise = np.random.randn(20) * 10
print(noise)

x = np.linspace(-5, 5, 20)
y = 10 * x - 7 + noise

plt.scatter(x, y)
plt.show()

# Разобьем данные на тренировочный и тестовый набор
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Давайте посмотрим на данные
plt.figure(figsize=(10, 5))
plt.plot(x, 10 * x - 7, label='real')
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_test, y_test, label='test')
plt.legend()
plt.show()

# Давайте создадим модель и обучим ее. Но перед этим, нам надо немного преобразовать данные.
# На вход модели мы должны подавать матрицу, а сейчас мы имеем вектор.

print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
print(x_train.shape)
print(x_test.shape)

print(x_train)
print(x_test)

model = LinearRegression()
model.fit(x_train, y_train)
# Теперь мы можем сделать предикт.
predictions = model.predict(x_test)
print(predictions)

# Так же можем посмотреть коэффеценты.
k = model.coef_
b = model.intercept_
print(k, b)

plt.figure(figsize=(10, 5))
plt.plot(x, 10 * x - 7, label='real')
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_test, y_test, label='test')
plt.plot(x, x * k + b, label='predicted')
plt.legend()
plt.show()

#########################
# Метрики
#########################
# Метрик довольно много все они хранятся в sklearn.metrics

y_train_predicted = mean_squared_error(model.predict(x_train), y_train)
y_test_predicted = mean_squared_error(model.predict(x_test), y_test)

print('Train MSE: ', y_train_predicted)
print('Test MSE: ', y_test_predicted)

y_train_predicted_mae = mean_absolute_error(model.predict(x_train), y_train)
y_test_predicted_mae = mean_absolute_error(model.predict(x_test), y_test)

print('Train MAE: ', y_train_predicted_mae)
print('Test MAE: ', y_test_predicted_mae)