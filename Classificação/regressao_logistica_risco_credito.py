#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:23:17 2019

@author: bruno
"""

import pandas as pd

base = pd.read_csv('risco-credito2.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores, classe)
print(classificador.intercept_)

# historia boa, divida alta, garantia nenhuma, renda > 35
# historia ruim, divida alta, garantia adqueada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)