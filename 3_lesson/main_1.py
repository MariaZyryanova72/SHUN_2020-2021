# NumPy — это мощный инструмент для работы с массивами.
import numpy as np

# Создадим массив. При его создании мы также можем задать тип данных, из которого он будет состоять.
a = np.array([0.1, 2, 1], dtype=np.float64)
print(a)

# Мы можем очень легко преобразовывать тип данных элементов массива.
print(a.dtype)
print(a.astype(int))
print(a.astype(str))

# Одномерные массивы

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# Считаем разные статистики
print(a.std(), a.sum(), a.prod(), a.min(), a.max(), a.mean(), sep='\n')

# Можно производить разные математические операции между массивом и числом. В результате получим массив.
print(a, a + 2, a - 2, a * 2, a / 2, a // 2, a % 2, a ** 2, sep='\n')

# Тоже самое можно сделать и с двумя массивами, тогда он будет производить эти операции поэлементно.
print(a + a, a - a, a * a, a / a, a // a, a % a, a ** a, sep='\n')

print(np.exp(a), np.sin(a), np.cos(a), np.round(a), sep='\n')

# один способ создать массив
a = np.arange(10, -1, -1)
print(a)
# Методы для изменения массива похожи с list
print(np.append(a, -1))
print(np.insert(a, 0, 11))
print(np.delete(a, [5, 7]))

# !!!!!!!! Методы для изменения массива(append, delete, insert) не меняют сам массив, а возвращают новую копию массива.

b = [1, 2, 3, 4]
b.append(5)
print(b)
print(id(a))
print(id(np.append(a, -1)))
print(id(np.insert(a, 0, 11)))
print(id(np.delete(a, [5, 7])))

# Многомерные массивы

# С многомерными массивами работает все то же, что и с одномерными, только еще больше
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)

print(np.append(a, -1))
print(np.insert(a, 0, 11))
print(np.insert(a, 0, 11, axis=0))
print(np.insert(a, 0, 11, axis=1))
print(np.delete(a, [5, 7]))

# Медоты для того, чтоб узнать размеры массива
print(a.shape, a.ndim, a.size, len(a), sep='\n')

# Изменение размерности массива
#
# Первый метод — это reshape. На вход этот медод принимает новые размеры массива

print(a)
print()
# reshape
print(a.reshape(9, 1))
print()
# вместо одной из осей можно просто вставить -1, тогда numpy попытается сам понять, какое там должно быть число
# если такое число не получится найти, то будет ошибка
print(a.reshape(9, -1))
print()
print(a.reshape(-1))

# Если невозможно преобразовать, то он выдаст ошибку
# print(a.reshape(2, 3))

# Так же мы можем выпрямить массив, для этого используется метод flatten
print(a.flatten())
print(a)

# Так же в многомерном массиве мы можем посчитать статистику по осям
print(a.std(axis=0), a.sum(axis=0), a.prod(axis=0), a.min(axis=0), a.max(axis=0), a.mean(axis=0), sep='\n')
print(a.std(axis=1), a.sum(axis=1), a.prod(axis=1), a.min(axis=1), a.max(axis=1), a.mean(axis=1), sep='\n')

# Крутой генератор NumPy
for i in np.arange(0, 1, 0.1):
    print(i)

for i in np.arange(0, 1, 0.1):
    print(round(i, 2))

# Но главным преимуществом NumPy является вот что
"""%time np.arange(0, 50000000)
%time list(range(0, 50000000))
print()"""

# Мы можем брать срезы. как делаем это в списках
a = np.arange(10000).reshape(100, 100)
print(a.shape, a[:10, :10], a[10:21, 30:41])

# Так же с NumPy очень просто создать стандартные массивы
print("Единичная матрица:\n", np.eye(5))
print("Матрица, состоящая из одних единиц:\n", np.ones((7, 5)))

# А еще можно легко транспонировать матрицу
print(a[:10, :10])
print(a[:10, :10].T)
