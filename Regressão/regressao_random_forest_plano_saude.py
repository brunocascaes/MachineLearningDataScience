#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:28:58 2019

@author: bruno
"""

import pandas as pd
base = pd.read_csv('house-prices.csv')
x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(x, y)
score = regressor.score(x, y)

import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x_teste, regressor.predict(x_teste), color='red')
plt.title("Regressao")
plt.xlabel('idade')
plt.ylabel('custo')