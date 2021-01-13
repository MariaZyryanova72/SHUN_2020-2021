import pandas as pd
from os import path

# Задача
# Расчитать среднее время жизни пользователя


ratings = pd.read_csv(path.join('data', 'ratings.csv'))

# userId — идентификатор пользователя
# movieId — идентификатор фильма
# rating — рейтинг фильма
# timestamp — время выставления рейтинга (количество секунд, прошедшее с 1 января 1970 года)


# Функция
# DataFrame.agg(self, func, axis=0, *args, **kwargs)
# позволяет проводить агрегацию из нескольких операций над заданной осью.
# В качестве параметров функция получает **kwargs, которые представляют собой столбец,
# над которым производится операция и собтсвенно имя функции.
# То же решение с применением лямбда выражений выглядит гораздо более лаконично и просто, мы делаем с
# помощью словаря и разделяем на max и min

mean_time_life = ratings.groupby('userId').agg({'timestamp': ['min', 'max']}).reset_index()
print(mean_time_life.head(2))

mean_time_life['lifetime'] = (
        (mean_time_life['timestamp']['max'] - mean_time_life['timestamp']['min']) / (60 * 60 * 24)).apply(round)
# Рассчитываем время жизни пользователя и записываем с столбец - lifetime и округляем
print(mean_time_life.head())

# Чтобы посчитать среднее время жизни пользователя восспользуемся mean
print(int(mean_time_life['lifetime'].mean()))
