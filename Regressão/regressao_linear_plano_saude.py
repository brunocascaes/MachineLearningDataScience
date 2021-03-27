import pandas as pd

base = pd.read_csv('plano-saude.csv')

x = base.iloc[:, 0].values
y = base.iloc[:, 1].values

import numpy as np
correlacao = np.corrcoef(x, y)

x = x.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

#b0
regressor.intercept_
#b1
regressor.coef_

#previsao1 = regressor.predict(np.array(40).reshape(1, -1))
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color='red')
plt.title('Regressao linear simples')
plt.xlabel('idade')
plt.ylabel('custo')

previsao1 = regressor.predict(np.array(40).reshape(1, -1))
previsao2 = regressor.intercept_ + regressor.coef_ * 40

score = regressor.score(x, y) # buscar valores maiores que 0.86

from yellowbrick.regressor import  ResidualsPlot # erro
visualizador = ResidualsPlot(regressor)
visualizador.fit(x, y)
visualizador.poof()
