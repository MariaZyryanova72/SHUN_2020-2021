# Объединение датасетов с помощью метода concat
from os import path, listdir
import pandas as pd

# Если merge приклеивает другой датасет сбоку(добавляет к датасету столбцы из другого датасета),
# то метод concat "приклеивает" другой датасет снизу(добавляет строки из другого датасета).
# Обязательным условием для метода concat является одинаковое количество и имена столбцов целевых датасетов.

path_files = path.join('data', 'ratings')
files = listdir(path_files)
print(files)
print(path.join(path_files, files[0]))

# Давайте посмотрим, из чего состоят наши файлы ratings

print(pd.read_csv(path.join(path_files, files[0])).head())
# Как видите, в них 4 столбика, но у них нет имени, поэтому первую строку с данными,
# Pandas считает за название столбцов.

# Задали имена столбцам
temp = pd.read_csv(path.join(path_files, files[0]),
                   names=['userId', 'movieId', 'rating', 'time'])
print(temp.head())

# Создадим целевой датасет, к которому будем приклеивать остальные таблицы.
data = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'time'])
data = pd.concat([data, temp])  # склеим их с помощью медота concat
data.reset_index().info()
print(data.head())

# приклеим все таблицы. Перед циклом пересоздадим, чтоб отменить предыдущее приклеивание.

data = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'time'])
for file in files:
    temp = pd.read_csv(path.join(path_files, file),
                       names=['userId', 'movieId', 'rating', 'time'])
    data = pd.concat([data, temp], ignore_index=True)
data.info()
# В результате получаем датацет, состоящий из 10 таблиц

