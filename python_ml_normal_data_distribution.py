# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:19:55 2023

@author: Marko
"""

import numpy
import matplotlib.pyplot as plt
# нормальний роподіл даних
# вказується, що середнє значення дорівнює 5.0
# стандартне відхилення дорівнює 1.0
x = numpy.random.normal(5.0,1.0,100000)
plt.hist(x,100)
plt.show()