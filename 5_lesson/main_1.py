# Нейронки
# Нейронки мы будем писать с помощью библиотеки tensorflow,
# с помощью ее надстройки keras.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#########################
# Перцептрон
#########################
#
# Персептрон (Perceptron) — простейший вид нейронных сетей.
# В основе лежит математическая модель восприятия информации мозгом,
# состоящая из сенсоров, ассоциативных и реагирующих элементов.

model = Sequential([
    Dense(2, input_dim=2, use_bias=False),
    Dense(1, use_bias=False)
])
model.summary()
weights = model.get_weights()
print(weights)  # посмотрим на веса нашей модели.

w1 = 0.42  # установиваем новые значения весов в ручную
w2 = 0.15
w3 = -0.56
w4 = 0.83
w5 = 0.93
w6 = 0.02
new_weight = [np.array([[w1, w3], [w2, w4]]), np.array([[w5], [w6]])]
print(new_weight)
model.set_weights(new_weight)

# Создадим тренировочные данные для обучения.
x_train = np.array([[7.2, -5.8]])
print(x_train.shape)
# И подадим эти данные на вход нейронной сети.
model.predict(x_train)  # По сути, нейросеть является просто функцией.

n1 = w1 * x_train[0][0] + w2 * x_train[0][1]  # Получим результат первого слоя.
n2 = w3 * x_train[0][0] + w4 * x_train[0][1]
print(n1)
print(n2)
# Получим результат на выходе
output = n1 * w5 + n2 * w6
print(output)
print(model.predict(x_train))


#################
# Функции активации
#################

# Но возникает проблема. Если все нейросети будут похожи на нашу,
# то нет смысла в большом количестве слоев из-за того, что, по сути,
# наша нейронная сеть будет обычной функцией. Поэтому придумали такие штуки,
# как функции активации. Именно они решают, какие нейроны следующего слоя будут активированны.


#################
# Sigmoid
#################

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


sigmoid_model = Sequential([
    Dense(2, input_dim=2, activation='sigmoid', use_bias=False),
    Dense(1, activation='sigmoid', use_bias=False)
])

sigmoid_model.set_weights(new_weight)
sigmoid_model.predict(x_train)

n1 = sigmoid(w1 * x_train[0][0] + w2 * x_train[0][1])
n2 = sigmoid(w3 * x_train[0][0] + w4 * x_train[0][1])
print(n1)
print(n2)

sigmoid_output = sigmoid(n1 * w5 + n2 * w6)
print(sigmoid_output)


#################
# Relu
#################

def relu(x):
    return np.clip(x, 0, np.inf)


relu_model = Sequential([
    Dense(2, input_dim=2, activation='relu', use_bias=False),
    Dense(1, activation='relu', use_bias=False)
])

relu_model.set_weights(new_weight)
relu_model.predict(x_train)

n1 = relu(w1 * x_train[0][0] + w2 * x_train[0][1])
n2 = relu(w3 * x_train[0][0] + w4 * x_train[0][1])
print(n1)
print(n2)

relu_output = relu(n1 * w5 + n2 * w6)
print(relu_output)
