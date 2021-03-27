#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:56:19 2019

@author: bruno
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
# gerador automatico de registros ALEATÓRIOS
# sempre os mesmos random_state = 0
x, y = make_blobs(n_samples = 200, centers = 4)
# y sao as definicoes de qual cluster esta associada ao reg
plt.scatter(x[:,0], x[:,1])

kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)

previsoes = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c = previsoes)