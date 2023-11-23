# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:21:18 2023

@author: Marko
"""

# KNN — це простий алгоритм машинного навчання (ML), який можна використовувати для завдань класифікації або регресії, а також часто використовується для імпутації відсутнього значення. Він заснований на ідеї, що спостереження, найближчі до певної точки даних, є найбільш «схожими» спостереженнями в наборі даних, і тому ми можемо класифікувати непередбачені точки на основі значень найближчих існуючих точок. Вибравши K , користувач може вибрати кількість найближчих спостережень для використання в алгоритмі.
import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

plt.scatter(x, y, c=classes)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
data = list(zip(x,y))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data,classes)

new_x = 8
new_y = 21
new_point = [(new_x,new_y)]
prediction = knn.predict(new_point)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x = new_x-1.7, y =new_y-0.7, s=f'new point, class: {prediction[0]}')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data,classes)
prediction = knn.predict(new_point)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x = new_x-1.7, y =new_y-0.7, s=f'new point, class: {prediction[0]}')
plt.show()
