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
