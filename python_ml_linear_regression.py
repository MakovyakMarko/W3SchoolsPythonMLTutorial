# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:31:00 2023

@author: Marko
"""

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

from scipy import stats
slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()
# коефіцієнт кореляції від 1 до -1, де 0 - відсутність кореляції
# а 1 чи -1 повна кореляція
print(r)
# передбачення швидкості 10 річної машини
speed = myfunc(10)
print(speed)

# приклад, де лінійна регресія не підходить

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
# розрахунок коефіцієнта кореляції:
print(r)