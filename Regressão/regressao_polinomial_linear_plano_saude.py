#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:23:48 2019

@author: bruno
"""

import pandas as pd
import numpy as np

base = pd.read_csv('plano-saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

# regressao linear simples
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x, y)
score1 = regressor1.score(x, y)

# previsao de quem tem 40 anos
#regressor1.predict(40)
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor1.predict(x), color = 'red')
plt.title('Regressão Linear')
plt.xlabel('Idade')
plt.ylabel('Custo')