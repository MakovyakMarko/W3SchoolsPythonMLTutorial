# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:11:35 2023

@author: Marko
"""

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)
# Створюємо набір даних
x = numpy.random.normal(3,1,100)
y = numpy.random.normal(150,40,100)/x
plt.scatter(x, y)
plt.show()
# Розділяємо на навчання/тест (80:20)
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]
# Відображаємо набір для навчання
plt.scatter(train_x, train_y)
plt.show()
# Відображаємо набір для тесту
plt.scatter(test_x, test_y)
plt.show()
# Підбираємо набір даних - поліноміальна регресія
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0,6,100)
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
# Перевіряємо, наскільки навчальні дані відповідають
# поліноміальній регресії 
r2 = r2_score(train_y, mymodel(train_x))
print(r2)
# Перевіряємо модель за допомогою даних тестування
r2 = r2_score(test_y, mymodel(test_x))
print(r2)
# Передбачаємо значення
print(mymodel(5))