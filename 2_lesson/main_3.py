# Методы matplotlib
from os import path
import pandas as pd
# Визуализировать данные пандас мы можем как с помощью методов самого Pandas, так и с помощью Matplotlib
import matplotlib.pyplot as plt

# Типы графиков в Matplotlib такие же, как в визуальзации Pandas
# Но в Matplotlib больше возможностей. Напрмер мы можем как создать
# одну область для графика и нарисовать в ней несколько графиков, так
# и нарисовать несколько графиков по отдельности. Заголовки, подписи осей,
# сетка и все отсальные вещи включаются и изменяются с помощью методов Matplotlib

data = pd.read_csv(path.join('data', 'tips.csv'))

fig = plt.figure()
axes_base = fig.add_axes([0, 0, 1, 1])
axes_base.hist(data['total_bill'], bins=50, color='green')
axes_base.set_title('Сумма')
axes_add = fig.add_axes([0.5, 0.5, 0.5, 0.5])
axes_add.scatter(x=data['total_bill'], y=data['tip'])
plt.show()

# Так и создать несколько гарфиков по-отдельности
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0][0].hist(data['total_bill'], bins=50, color='green')
axes[1][0].hist(data['tip'], bins=20, color='red')
axes[0][1].scatter(x=data['total_bill'], y=data['tip'])
axes[1][1].scatter(x=data['tip'], y=data['tip'])
plt.show()
