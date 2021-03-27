import pandas as pd

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()  

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split 
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
        previsores, classe, test_size=0.25, random_state=0)

from sklearn.svm import SVC
classificador = SVC(kernel='rbf', random_state=1, C=0.25, gamma='auto')
# linear / poly / rbf / etc
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste) # resultado da previsao

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes) # percentual de acerto
matriz = confusion_matrix(classe_teste, previsoes)