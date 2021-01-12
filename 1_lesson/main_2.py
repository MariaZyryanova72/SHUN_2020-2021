import pandas as pd
from os import path


# Задача
# написать гео-классификатор, который каждой строке сможет выставить
# географическую принадлежность определенному региону. Т. е. если поисковый
# запрос содержит название города региона, то в столбце 'region' пишется название
# этого региона. Если поисковый запрос не содержит названия города, то ставим 'undefined'

def geo_classification(keyword):
    for region, cities in geo_data.items():
        if any(x in keyword for x in cities):
            return region
        else:
            return 'undefined'


data = pd.read_csv(path.join('data', 'keywords.csv'))

geo_data = {
    'Центр': ['москва', 'тула', 'ярославль'],
    'Северо-Запад': ['петербург', 'псков', 'мурманск'],
    'Дальний Восток': ['владивосток', 'сахалин', 'хабаровск']
}

data['region'] = data['keyword'].apply(geo_classification)


print(data[data['region'] != 'undefined'].head())  # Выводим все запросы, которые не пустые
