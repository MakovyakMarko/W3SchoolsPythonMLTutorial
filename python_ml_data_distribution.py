# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:49:56 2023

@author: Marko
"""

# створіть масив, що містить 250 випадкових чисел з 
# плаваючою точкою від 0 до 5:

import numpy 
x = numpy.random.uniform(0.0,5.0,250)
print(x)

import matplotlib.pyplot as plt
plt.hist(x,5)
plt.show()

x = numpy.random.uniform(0.0,5.0,100000)
plt.hist(x,100)
plt.show()