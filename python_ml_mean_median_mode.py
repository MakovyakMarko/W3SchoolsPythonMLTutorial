# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:09:34 2023

@author: Marko
"""

# mean - середня швидкість (сума всіх чисел розділена на кількість значень)
import numpy 
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.mean(speed)
print("mean:",x)
# median - значення посереднині, після того як все відсортовано
# якщо всередині 2 числа, то поділить їх суму на 2
x = numpy.median(speed)
print("median:",x)
# mode - значення, яке зустрічається найбільшу кількість разів
from scipy import stats
x = stats.mode(speed)
print("mode:", x)

# mean, median, mode - це методи, які часто використовуються в 
# машинному навчанні, тому важливо розуміти концепцію, що стоїть за ними