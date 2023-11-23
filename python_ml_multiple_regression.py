# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:57:31 2023

@author: Marko
"""

import pandas
from sklearn import linear_model
# передбачаємо викид СО2 для авто вагою 2300 кг і об'ємом 
# двигуна 1.3 літра. Відповідно до даних в датасеті
df = pandas.read_csv("data.csv")

X = df[['Weight',"Volume"]]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300,1300]])

print(predictedCO2)
# Розрахуємо коефіцієнт ваги та об'єму стосовно СО2
print(regr.coef_)
# Розрахуємо, чи вірно вказаний коефіцієнт ваги 0.00755095
# Добавимо 1000 кг, має збільшитись до 114
predictedCO2 = regr.predict([[3300,1300]])

print(predictedCO2)
