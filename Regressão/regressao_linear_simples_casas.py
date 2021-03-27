import pandas as pd

base = pd.read_csv('house-prices.csv')
x = base.iloc[:, 5:6].values # formato do numpy
y = base.iloc[:, 2:3].values

#divide a base de dados em 70% treinamento e 30% teste
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treinamento, y_treinamento)
score = regressor.score(x_treinamento, y_treinamento)
# deu 0.49, ou seja, ruim // nao se adaptou mto bem só com 1 atributo

import matplotlib.pyplot as plt
plt.scatter(x_treinamento, y_treinamento)
plt.plot(x_treinamento, regressor.predict(x_treinamento), color = 'red')
# nao é um modelo mto bom

previsoes = regressor.predict(x_teste)

# calculo do erro
resultado = abs(y_teste - previsoes)
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes) # mais utilizado (raiz)

# com teste
plt.scatter(x_teste, y_teste)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
regressor.score(x_teste, y_teste)