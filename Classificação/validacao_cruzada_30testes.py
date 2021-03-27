import pandas as pd

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores, # transforma em array para nao perder os indices
                                                    np.zeros(shape=(previsores.shape[0], 1))):
        classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) # passa os indices de treinamento para p classificador
        previsoes = classificador.predict(previsores[indice_teste])         # preve com o indice de teste
        precisao = accuracy_score(classe[indice_teste], previsoes)          # na precisao passa as previsoes e a classe (teste)
        resultados1.append(precisao)                                        # adiciona na lista
    resultados1 = np.asarray(resultados1) # faz a transformacao para poder fazer a media
    media = resultados1.mean()
    resultados30.append(media) # adiciona na lista o resultado
resultados30 = np.asarray(resultados30) #transforma em vetor
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))
