#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:09:17 2019

@author: bruno
"""
# agrupamento com mais atributos
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit-card-clients.csv', header = 1)
#cria nova coluna
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] +base['BILL_AMT5'] + base['BILL_AMT6']
x = base.iloc[:, [1,2,3,4,5,25]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) # distancia
plt.plot(range(1,11) , wcss)
plt.xlabel('Ńúmero de clusters')
plt.ylabel('WCSS')

kmeans = KMeans(n_clusters = 4, random_state = 0)
previsoes = kmeans.fit_predict(x)

lista_clientes = np.column_stack([base, previsoes])
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]