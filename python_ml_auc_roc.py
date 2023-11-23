# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:48:54 2023

@author: Marko
"""

# Припустімо, що ми маємо незбалансований набір даних, де більшість наших даних має одне значення. Ми можемо отримати високу точність моделі, прогнозуючи мажоритарний клас.
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score, roc_curve
n = 10000
ratio = .95
n_0 = int((1-ratio)*n)
n_1 = int(ratio * n)
y = np.array([0] * n_0 + [1] * n_1)
# below are the probabilities obtained from a hypothetical model that always predicts the majority class
# probability of predicting class 1 is going to be 100%
y_proba = np.array([1]*n)
y_pred = y_proba > .5
print(f'accuracy score: {accuracy_score(y,y_pred)}')
cf_mat = confusion_matrix(y, y_pred)
print("Confusion matrix")
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')
# Хоча ми отримуємо дуже високу точність, модель не надає інформації про дані, тому вона не корисна. Ми точно прогнозуємо клас 1 у 100% випадків, тоді як неточно прогнозуємо клас 0 у 0% випадків. За рахунок точності, можливо, було б краще мати модель, яка могла б дещо розділити два класи.
# below are the probabilities obtained from a hypothetical model that doesn't always predict the mode
y_proba_2 = np.array(
    np.random.uniform(0, .7, n_0).tolist()+
    np.random.uniform(.3, 1, n_1).tolist()
)
y_pred_2 = y_proba_2 > .5
print(f'accuracy score: {accuracy_score(y,y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print("Confusion matrix")
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

# Для другого набору прогнозів ми не маємо такого високого показника точності, як для першого, але точність для кожного класу більш збалансована. Використовуючи точність як показник оцінки, ми б оцінили першу модель вище, ніж другу, навіть якщо вона нічого не говорить нам про дані.

# У таких випадках краще використовувати інший показник оцінки, наприклад AUC.

import matplotlib.pyplot as plt
def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """
    fpr,tpr,thresholds = roc_curve(true_y,y_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
# model 1    
plot_roc_curve(y, y_proba)
print(f'model 1 AUC score: {roc_auc_score(y,y_proba)}')
# model 2
plot_roc_curve(y, y_proba_2)
print(f'model 2 AUC score: {roc_auc_score(y,y_proba_2)}')

n = 10000
y = np.array([0] * n + [1] * n)
y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist()+
    np.random.uniform(.3, .7, n).tolist()+
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist()+
    np.random.uniform(.3, .7, n).tolist()+
    np.random.uniform(.6,1,n//2).tolist()
)
print(f'model 1 accuracy score: {accuracy_score(y,y_prob_1>.5)}')
print(f'model 2 accuracy score: {accuracy_score(y,y_prob_2>.5)}')
print(f'model 1 AUC score: {roc_auc_score(y,y_prob_1)}')
print(f'model 2 AUC score: {roc_auc_score(y,y_prob_2)}')
# plot model 1
plot_roc_curve(y,y_prob_1)
# plot model 2
fpr,tpr,thresholds = roc_curve(y,y_prob_2)
plt.plot(fpr,tpr)