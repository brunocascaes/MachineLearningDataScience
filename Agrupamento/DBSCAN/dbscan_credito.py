import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit-card-clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] +base['BILL_AMT5'] + base['BILL_AMT6']

x = base.iloc[:,[1,25]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

dbscan = DBSCAN(eps = 0.37, min_samples = 4)
previsoes = dbscan.fit_predict(x)
unicos, quantidade = np.unique(previsoes, return_counts = True)
# quantidade e valores unicos

plt.scatter(x[previsoes == 0, 0], x[previsoes == 0, 1], s= 100, c='red', label = 'Cluster 1')
# somente as linhas que fazem parte do cluster 0 na coluna 0 // cluster 0 na coluna 1
plt.scatter(x[previsoes == 1, 0], x[previsoes == 1, 1], s= 100, c='orange', label = 'Cluster 2')
plt.scatter(x[previsoes == 2, 0], x[previsoes == 2, 1], s= 100, c='green', label = 'Cluster 3')
plt.scatter(x[previsoes == 3, 0], x[previsoes == 3, 1], s= 100, c='blue', label = 'Cluster 4')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]