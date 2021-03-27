#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:20:25 2019

@author: bruno
"""

import pandas as pd
base = pd.read_csv('plano-saude2.csv')
x = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x, y)
score = regressor.score(x, y)

import matplotlib.pyplot as plt
plt.scatter(x, y) # plota x e y
plt.plot(x, regressor.predict(x), color = 'red') # perfeito

import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1, 1) # n muda linha
plt.scatter(x, y) # plota x e y
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
#desenho correto, pois arvores não são um modelo continuo
#cada ponto é considerado um split