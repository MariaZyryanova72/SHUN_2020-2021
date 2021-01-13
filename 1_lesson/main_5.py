import pandas as pd
from os import path
import numpy as np

# Задача
# определить самые популярные жанры


def genres_ratings(data):
    return pd.Series([data['rating'] if genre in data['genres'] else np.NaN for genre in genres])


genres = ['Drama', 'Action', 'Thriller', 'Comedy', 'Romance', 'War', 'Mystery', 'Crime']
ratings = pd.read_csv(path.join('data', 'ratings.csv'))
movies = pd.read_csv(path.join('data', 'movies.csv'))

joined = ratings.merge(movies, on='movieId', how='left')  # Объедениение двух df
print(joined.head())

# Основные параметры метода merge:
# how — при значении left берем все значения из ratings и
# ищем их соответствия в movies. Если нет совпадений, то ставим
# нулевое значение. При этом все значения из ratings сохраняются.
# Другие варианты: right, inner (оставляем только те movieId, которые
# есть в обоих датафреймах), outer (объединение всех вариантов movieId в датафреймах).
#
# on — по какому столбцу происходит объединение.
# Для объединения по нескольким столбцам используйте
# on = ['col1', 'col2'] или left_on и right_on.

joined[genres] = joined.apply(genres_ratings, axis=1)
# Применяем функцию genres_ratings ко всем строкам - 1 (0 - столбцы)
print(joined.head(3))

# если 0
for genre in genres:
    print(f'{genre} mean rating {joined[genre].mean():.2f}')

# если NaN
for genre in genres:
    print(f'{genre} mean rating {joined[genre].mean():.2f}')


"""
# Опасность merge

ratings = pd.read_csv(path.join('data', 'ratings_example.csv'), sep = '\t')
movies = pd.read_csv(path.join('data', 'movies_example.csv'), sep = '\t')

# Могут появиться дубли
ratings.merge(movies, how = 'left', on = 'movieId')

# Но это легко решить
movies.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)
ratings.merge(movies, how = 'left', on = 'movieId')

# Как left, right, inner и outer влияют на объединение
# inner - x /\ y
# outer - x \/ y
# left - x => y
# right - x <= y

ratings.merge(movies, how = 'left', on = 'movieId')
ratings.merge(movies, how = 'right', on = 'movieId')
ratings.merge(movies, how = 'inner', on = 'movieId')
ratings.merge(movies, how = 'outer', on = 'movieId')
"""
