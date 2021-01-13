import pandas as pd
from os import path

# Задача
# проверить верно ли, что с ростом года выпуска фильма его средний рейтинг становится ниже


def production_year(row):
    for year in years:
        if year in row['title']:
            return year
    return '1900'


years = [str(i) for i in range(1950, 2019)]

ratings = pd.read_csv(path.join('data', 'ratings.csv'))
movies = pd.read_csv(path.join('data', 'movies.csv'))
joined = ratings.merge(movies, on='movieId', how='left')

joined['year'] = joined.apply(production_year, axis=1)
joined[['year', 'rating']].groupby('year').mean().sort_values('rating', ascending=False)

joined[['year', 'rating']].groupby('year').mean().sort_values('rating', ascending=False).plot()

joined.to_csv('filename.csv')
joined.to_excel('filename.xlsx')
table = pd.read_html('http://kraka-race.ru/page-44.html')
print(table)
print(help(pd.read_html))