#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:58:23 2019

@author: bruno
"""

import pandas as pd
dados = pd.read_csv('mercado.csv', header = None) # sem cabeçalho

# apyori recebe lista e nao dataframe
transacoes = []
for i in range(0, 10): # 10 registros
    transacoes.append([str(dados.values[i, j]) for j in range(0,4)])
# length, quantidade minima de "produtos"
from apyori import apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2

resultadoFormatado = []
for j in range(0, 3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado