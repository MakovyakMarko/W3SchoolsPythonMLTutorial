# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:24:41 2023

@author: Marko
"""
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# Читаємо набір даних за допомогою pandas
import pandas
df = pandas.read_csv('data_tree.csv')
print(df)
# Перетворюємо всі дані в числові, це потрібно для створення 
# дерева рішень. Використовуємо для цього метод pandas - map()
d = {'UK':0,'USA':1,'N':2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES':1,'NO':0}
df['Go'] = df['Go'].map(d)
print(df)
# Відокремлюємо стовпці функцій від цільового стовпця
# Стовпці функцій - стовпці, за якими ми намагаємось передбачити
# Цільовий стовпець - стовпець із значеннями, які ми намагаємось передбачити
features = ['Age','Experience','Rank','Nationality']
X = df[features]
y = df['Go']
print(X)
print(y)
# Створюємо дерево рішень
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)
tree.plot_tree(dtree, feature_names=features)
# Передбачення значень
# чи варто мені піти на шоу, де грає 40-річний американський
# комік, із 10-річним стажем і ретингом комедії 7 
print(dtree.predict([[40,10,7,1]]))
# а якщо рейтинг 6?
print(dtree.predict([[40,10,6,1]]))