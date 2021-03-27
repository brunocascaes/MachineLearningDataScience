#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:19:56 2019

@author: bruno
"""

import pandas as pd

# TRATAMENTO DE VALORES INCONSISTENTES
base = pd.read_csv('credit-data.csv')   # carrega dataset
base.describe()                         # estatisticas
base.loc[base['age'] < 0]               # localiza no dataset
base.drop('age', 1, inplace=True)       # apaga coluna 
base.drop(base[base.age < 0])           # apaga registros com problema
base['age'].mean()                      # média com todos registros
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()  
# substitui os registros com problema pela média


# ENCONTRANDO VALORES FALTANTES
pd.isnull(base['age'])                  # Retorna true ou false
base.loc[pd.isnull(base['age'])]        # Retorna os registros

features = base.iloc[:, 1:4].values
# PREVISORES = FEATURES
# iloc - Divide os dados do dataset
# : - pega todas as linhas
# 1:4 - pega as colunas 1, 2 e 3

label = base.iloc[:, 4].values
#CLASSE - LABEL

import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(features[:, 0:3])
features[:, 0:3] = imputer.transform(features[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)