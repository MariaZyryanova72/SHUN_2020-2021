import pandas as pd
from os import path

# Задача
# получить для каждого пользователя распределение по количеству выставленных оценок


ratings = pd.read_csv(path.join('data', 'ratings.csv'))

# Создадим сводную даблицу, где для каждого пользователя будет количество каждых оценок, которые он выставлял
print(ratings['rating'][0])
estimate_count = ratings.pivot_table(index='userId', columns='rating', values='timestamp', aggfunc='count')
print(estimate_count.head())

estimate_count = estimate_count.fillna(0).reset_index(level='userId')  # Убираем NaN с помощью метода fillna
print(estimate_count.head())
