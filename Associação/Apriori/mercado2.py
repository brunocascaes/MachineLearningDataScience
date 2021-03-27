#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:23:02 2019

@author: bruno
"""

import pandas as pd
dados = pd.read_csv('mercado2.csv') # baixar o dataset
transacoes = []
for i in range(0, 7501): # demora um bocado
    transacoes.append([str(dados.values[i, j]) for j in range(0, 20)])

from apyori import apriori
regras = apriori(transacoes, min_support = 0.003, min_confidence = 0.2, min_lift = 2.0, min_length = 2)
# passa o dataset em formato de lista, o minimo suporte, minima confiança e o minimo lift
# o que é o min_length

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2
resultadoFormatado = []
for j in range(0, 3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado