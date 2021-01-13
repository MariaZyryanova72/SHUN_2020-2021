# Библиотека SKLearn - библиотека для решения задач машинного обучения.
import numpy as np
from os import path
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error
# Нас интересует средняя квадратичная(MSE) и средняя абсолютная ошибка(MAE).
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics

sklearn.metrics.__dir__()

# Модуль, где находятся модели ближайших соседей для задач регрессии, так и для классификации.
from sklearn import neighbors

sklearn.neighbors.__dir__()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

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

#########################
# Метод k ближайших соседей
#########################
# kNN расшифровывается как k Nearest Neighbor или k Ближайших Соседей — это один из самых простых алгоритмов
# классификации, также иногда используемый в задачах регрессии.

# Задание: Предсказание сердечно-сосудистых заболеваний
#
# Данные возьмем с этого соревнования:
#
# https://mlbootcamp.ru/round/12/sandbox/


data = pd.read_csv(path.join('data', 'ml5', 'train.csv'), sep=';')
print(data.head())

"""В этом датасете данные о 70000 человек, о каждом из которых известно:
-- id человека (id) -- возраст человека в днях (age) -- пол (gender) -- рост в сантиметрах
 (height) -- вес в килограммах (weight) -- верхнее артериальное давление (ap_hi) -- 
 нижнее артериальное давление (ap_lo) -- показатель холестерина (cholesterol, 1, 2 или 3) 
 -- показатель глюкозы (gluc) -- курит ли человек (smoke, 0--не курит, 1--курит) -- употребляет
  ли человек алкоголь (alco, 0--нет, 1--да) -- ведет ли активную жизнь (active)

Целевая переменная: cardio -- наличие у человека сердечно-сосудистого заболевания. 1 -- есть, 0 -- нет."""

print(data.describe())  # Вся статистика датасета

# Подготовка данных
#
# Поделим на данные и целевую переменную

y = data['cardio']
data = data[list(data.columns[:-1])]
del data['id']
print(data.dtypes)  # все признаки числовые и все хорошо

# Приводим данные в читаемый вид
data_ch = pd.get_dummies(data, columns=['cholesterol'])
print(data_ch.head())
data_age = data.copy()
data_ch_age = data_ch.copy()

data_age["age"] /= 365
data_ch_age["age"] /= 365
print(data_age.head())

# Разобьем данные на тестовые и тренировочные
data_train, data_val, y_train, y_val = train_test_split(data, y, test_size=0.3)
data_ch_train, data_ch_val, y_ch_train, y_ch_val = train_test_split(data_ch, y, test_size=0.3)

data_age_train, data_age_val, y_age_train, y_age_val = train_test_split(data_age, y, test_size=0.3)
data_ch_age_train, data_ch_age_val, y_ch_age_train, y_ch_age_val = train_test_split(data_ch_age, y, test_size=0.3)

# Приступим к обучению
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train, y_train)

knn.predict_proba(data_val)
predicted_proba = knn.predict_proba(data_val)[:, 1]
print(predicted_proba)

print(log_loss(y_val, predicted_proba))  # Посчитаем метрику

# Так же классифицировать можно с помощью логистической регрессии.

model = LogisticRegression(max_iter=1000)
model.fit(data_train, y_train)
predicted_lr = model.predict(data_val)
print(predicted_lr)
print(log_loss(y_val, predicted_lr))

