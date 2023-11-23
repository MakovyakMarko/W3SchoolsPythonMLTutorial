# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:04:35 2023

@author: Marko
"""

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight','Volume']]

scaledX = scale.fit_transform(X)

print(scaledX)

y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])

print(predictedCO2)
