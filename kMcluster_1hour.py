import pandas as pd # This is always assumed but is included here as an introduction.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import openpyxl
import xlwt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
plt.style.use('seaborn')

haushalt=pd.read_excel('datasets/clustering_hour.xlsx')
print(haushalt)

haushalt.rename(columns={'Unnamed: 0': 'Zeit'}, inplace=True)
print(haushalt['Zeit'])
print(haushalt['cluster'])
print(haushalt)

plt.figure(figsize=(16, 5))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

i=0
j=0
print(haushalt['cluster'][3])
print(haushalt['Zeit'][3])
print(haushalt[0][1])
while i<8760:
    while j <74:
        if haushalt['cluster'][i] == 0:
            plt.scatter(haushalt.index[i], haushalt[j][i], color='red', s=2.2)

        if haushalt['cluster'][i] == 1:
            plt.scatter(haushalt.index[i], haushalt[j][i], color='blue', s=2.2)

        j=j+1

    print(str(haushalt.index[i]))
    i=i+1
    j=0

plt.savefig('datasets/kMeansClustering_1hour.png', bbox_inches='tight')
plt.show()