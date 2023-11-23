# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:16:29 2023

@author: Marko
"""

from sklearn import datasets
# Щоб краще зрозуміти CV, ми будемо використовувати різні методи на наборі даних райдужки. Давайте спочатку завантажимо та розділимо дані.
X, y = datasets.load_iris(return_X_y=True)
# Існує багато методів перехресної перевірки, ми почнемо з перегляду k-кратної перехресної перевірки.
# Оскільки ми намагатимемося класифікувати різні види квітів ірису, нам потрібно буде імпортувати модель класифікатора, для цієї вправи ми будемо використовувати DecisionTreeClassifier. Нам також потрібно буде імпортувати модулі резюме з sklearn.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
# Із завантаженими даними тепер ми можемо створити та адаптувати модель для оцінки.
clf = DecisionTreeClassifier(random_state=42)
# Тепер давайте оцінимо нашу модель і подивимося, як вона працює на кожному k -кратному рівні.
k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv = k_folds)
# Також доцільно побачити загальну ефективність CV шляхом усереднення балів для всіх складок.
print("Cross Validation Scores: ", scores)
print("Average CS Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
# Замість того, щоб вибирати кількість поділів у наборі навчальних даних, наприклад k-fold LeaveOneOut, використовуйте 1 спостереження для перевірки та n-1 спостереження для навчання. Цей метод є вичерпною технікою.
from sklearn.model_selection import LeaveOneOut, cross_val_score
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv = loo)
print("Cross Validation Scores: ", scores)
print("Average CS Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
# Leave-P-Out — це просто тонка відмінність від ідеї Leave-One-Out, оскільки ми можемо вибрати кількість p для використання в нашому наборі перевірки
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
scores = cross_val_score(clf, X, y, cv=lpo)
print("Cross Validation Scores: ", scores)
print("Average CS Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
# Shuffle Split
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores =cross_val_score(clf, X, y, cv = ss)
print("Cross Validation Scores: ", scores)
print("Average CS Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
