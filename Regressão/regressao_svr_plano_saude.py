#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:58:24 2019

@author: bruno
"""

import pandas as pd
base = pd.read_csv('plano-saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

#kernel linear, como se fosse a regressao normal
from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(x, y) # treina

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor_linear.predict(x), color = 'red')
regressor_linear.score(x, y)

# kernel poly
regressor_poly = SVR(kernel = 'poly', degree = 2) # grau
regressor_poly.fit(x, y)
plt.scatter(x, y)
plt.plot(x, regressor_poly.predict(x), color = 'blue')
regressor_poly.score(x, y)

# kernel poly
regressor_poly = SVR(kernel = 'poly', degree = 3) # grau
regressor_poly.fit(x, y)
plt.scatter(x, y)
plt.plot(x, regressor_poly.predict(x), color = 'black')
regressor_poly.score(x, y)

# kernel rbf - necessario fazer o escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(x, y)
plt.scatter(x, y)
plt.plot(x, regressor_rbf.predict(x), color = 'black')
regressor_rbf.score(x, y)