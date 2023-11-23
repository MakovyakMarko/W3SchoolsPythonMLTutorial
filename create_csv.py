# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:15:16 2023

@author: Marko
"""

import pandas as pd

data = {
        'Age':[36,42,23,52,43,44,66,35,52,35,24,18,45],
        'Experience':[10,12,4,4,21,14,3,14,13,5,3,3,9],
        'Rank':[9,4,6,4,8,5,7,9,7,9,5,7,9],
        'Nationality':['UK','USA','N','USA','USA','UK','N','UK','N','N','USA','UK','UK'],
        'Go':['NO','NO','NO','NO','YES','NO','YES','YES','YES','YES','NO','YES','YES']
        }

df = pd.DataFrame(data)
df.to_csv('data_tree.csv', index=False)