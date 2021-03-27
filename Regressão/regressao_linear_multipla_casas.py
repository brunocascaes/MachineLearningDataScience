#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:56:18 2019

@author: bruno
"""

import pandas as pd
base = pd.read_csv('house-prices.csv')
x = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treinamento, y_treinamento)
score = regressor.score(x_treinamento, y_treinamento) # 0.7

previsoes = regressor.predict(x_teste)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

regressor.score(x_teste,  y_teste) # 0.68

regressor.intercept_
regressor.coef_