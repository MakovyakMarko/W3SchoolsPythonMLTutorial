# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:02:58 2023

@author: Marko
"""

# Процентилі використовуються в статистиці, щоб дати вам 
# число, яке описує значення, за яке даний відсоток 
# значень нижчий.

import numpy 
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 75)
print(x)
# якого віку досягли 90 процентів людей?
x = numpy.percentile(ages, 90)
print(x)