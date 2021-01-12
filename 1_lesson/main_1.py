import pandas as pd
from os import path

# Задача:
# дана статистика поисковых запросов вашей потенциальной целевой аудитории за 11 дней,
# необходимо посчитать распределение количества слов в поисковых запросах этого файла
# на тему недвижимости, т. е. понять, сколько запросов содержали 1 слово, 2 слова и т. д.
# поисковый запрос с каким количеством слов встречается в нашем датасете чаще всего?


data = pd.read_csv(path.join('data', 'keywords.csv'))

# data.info()  # информация о данных

# посчитаем количество запросов в день
data['daily_shows'] = data['shows'] / 11
print(data.head(2))  # Выводим 2 строки

# Первым делом, нам надо найти все вопросы,
# связанные с недвижимостью. Давайте оставим только те запросы,
# в которых есть слова включающие в себя "недвиж".
print(data[data['keyword'].str.contains('недвиж')].head())

print(data[data['shows'] < 10000].head())  # Выводим только те запросы, в которых просмотров меньше 10 000

data['split'] = data['keyword'].str.split()  # разделим каждый запрос на отдельные слова
print(data.head())

data['words_count'] = data['split'].apply(len)  # найдем количество слов в каждом запросе
print(data.head())

# Метод groupby позволяет группировать датафрейм по каким либо столбам

data = data.groupby('words_count').sum().sort_values(by='shows', ascending=False)
print(data.head(10))

data = data.loc[:, ['shows', 'words_count']].groupby('words_count').agg(['min', 'max', 'count', 'sum', 'mean'])
print(data.head())

data = data.loc[:, ['shows', 'words_count']].groupby('words_count').agg(
    ['min', 'max', 'count', 'sum', 'mean']).reset_index()
print(data.head(10))

data = data.loc[:, ['shows', 'words_count']].groupby('words_count').agg(
    ['min', 'max', 'count', 'sum', 'mean']).reset_index()
print(data.head(10))

data = data[data['words_count'] % 2 == 0].reset_index()