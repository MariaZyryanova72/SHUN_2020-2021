# Визуализация
from os import path
import pandas as pd
# Визуализировать данные пандас мы можем как с помощью методов самого Pandas, так и с помощью Matplotlib
import matplotlib.pyplot as plt


data = pd.read_csv(path.join('data', 'tips.csv'))
print(data.head())

# Методы Pandas
data.plot()
plt.show()

data['total_bill'].plot()
plt.show()

data['total_bill'].plot(kind='hist', grid=True, title='Сумма')
plt.show()

data[['total_bill', 'tip']].plot(kind='hist',
                                 subplots=True,
                                 title=['Сумма', 'Чаевые'])
plt.show()

data.plot(x='total_bill',
          y='tip',
          kind='scatter')
plt.show()

data.pivot_table(values=['total_bill', 'tip'],
                 index='day',
                 aggfunc='mean').plot(kind='bar')
plt.show()


