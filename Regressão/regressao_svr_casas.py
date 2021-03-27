#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:19:32 2019

@author: bruno
"""

import pandas as pd
base = pd.read_csv('house-prices.csv')

x = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_treino, y_treino)
score = regressor.score(x_treino, y_treino) # 0.81 - boa

regressor.score(x_teste, y_teste) # todos escalonados ou todos nao

previsoes = regressor.predict(x_teste) # de maneira escalonada

# reinverter o escalonamentos
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)