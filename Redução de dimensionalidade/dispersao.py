import pandas as pd
base = pd.read_csv('credit-data.csv')
base = base.dropna()

#outliers idade
import matplotlib.pyplot as plt
plt.scatter(base.iloc[:,1], base.iloc[:,2])

#outliers loan
plt.scatter(base.iloc[:,1], base.iloc[:,3])

#agexloan
plt.scatter(base.iloc[:,2], base.iloc[:,3])