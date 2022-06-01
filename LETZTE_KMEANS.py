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

haushalt = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_123_days.xlsx')
print(haushalt)

timeindex = pd.date_range('00:00:00', '23:59:00', freq='1min')

#sc=MinMaxScaler()
#haushalt=sc.fit_transform(haushalt)

#kmeans=KMeans(n_clusters=3)
#cluster_found=kmeans.fit_predict(haushalt)
#cluster_found_sr=pd.Series(cluster_found, name='cluster')
#print(cluster_found_sr)

#cluster_found_sr.to_excel('datasets/HTW Berlin/123_clustering_oneday_clusters.xlsx')
plt.figure(figsize=(16, 5))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

i=0
j=0
cluster = np.arange(0, 3, step=1)
cluster_color = ['b','g','r']
index = 0
while i<1440:
    while j <1095:
        if haushalt['cluster'][i] == 0:
            plt.scatter(i, haushalt[j][i], color='b' ,s=3)
        if haushalt['cluster'][i] == 1:
            plt.scatter(i, haushalt[j][i], color='g', s=3)
        if haushalt['cluster'][i] == 2:
            plt.scatter(i, haushalt[j][i], color='r', s=3)

        print(str(haushalt.index[i])+' '+str(j))
        j=j+1
    #print(str(haushalt.index[i]))
    i = i + 1
    j = 0

plt.savefig('datasets/kMeansClustering_allinoneday_123.png', bbox_inches='tight')
plt.show()