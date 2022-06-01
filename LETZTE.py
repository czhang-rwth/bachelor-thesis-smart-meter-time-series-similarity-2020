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

def plotten_blau(x, y, title, xlabel, ylabel,datei, dpi=100):
    plt.figure(figsize=(16, 5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.plot(x, y, 's-',label=label)
    plt.plot(x, y,'s-')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')
    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(datei, bbox_inches='tight')
    plt.show()

def indexieren_htw_xlsx(csv_Datei ,excel_Datei):
    haushalt_csv = pd.read_csv(csv_Datei)
    ### Zwischenschritt print(haushalt_csv)
    haushalt_csv = haushalt_csv['Timestamp;PL1[W];PL2[W];PL3[W];QL1[W];QL2[W];QL3[W]'].str.split(';', expand=True)
    ### Zwischenschritt print(haushalt_csv)
    #haushalt_csv.to_excel(excel_Datei)
    haushalt = pd.read_excel(excel_Datei)
    len_0 = len(haushalt)
    ### Zwischenschritt print(haushalt)

    haushalt.rename(columns={'Unnamed: 0': 'Tage'}, inplace=True)
    haushalt.rename(columns={0: 'Zeit'}, inplace=True)
    haushalt.rename(columns={1: 'PL1[W]'}, inplace=True)
    haushalt.rename(columns={2: 'PL2[W]'}, inplace=True)
    haushalt.rename(columns={3: 'PL3[W]'}, inplace=True)
    haushalt.rename(columns={4: 'QL1[W]'}, inplace=True)
    haushalt.rename(columns={5: 'QL2[W]'}, inplace=True)
    haushalt.rename(columns={6: 'QL3[W]'}, inplace=True)

    #haushalt.index = pd.to_datetime(haushalt['Tage'] + haushalt['Zeit']) ###Fehler: index die ganze Zeitreihe wirkt danach nicht mehr
    haushalt['Zeitreihe'] = pd.Series((haushalt['Tage'] + haushalt['Zeit']), index=haushalt.index)
    haushalt['Wirkleistung PL[kW]'] = pd.Series((haushalt['PL1[W]'] + haushalt['PL2[W]'] + haushalt['PL3[W]']) * 0.001,
                                                index=haushalt.index)
    haushalt = haushalt.drop(columns=['Tage', 'Zeit','PL1[W]', 'PL2[W]', 'PL3[W]', 'QL1[W]', 'QL2[W]', 'QL3[W]'])
    ### Zwischenschritt print(haushalt)
    #print(haushalt['Wirkleistung PL[kW]'].dtypes)
    #haushalt[column] = haushalt[column].astype(int)
    index_neu = pd.date_range('01.01.2010', '01.01.2011', freq='1min', closed='left')
    haushalt_neu = pd.Series(np.arange(len_0), index=index_neu, dtype=float)
    #####haushalt_neu = pd.Series(haushalt['Wirkleistung PL[kW]'], index=index_neu, dtype=float) ###Fehler: haushalt_neu zeigt dann immer NaN
    i = 0
    while i < len_0:
        haushalt_neu[i] = haushalt['Wirkleistung PL[kW]'][i]
        ###print(i, " ", haushalt['Wirkleistung PL[kW]'][i]," ", haushalt_neu[i] )
        i=i+1

    print(haushalt_neu)
    return haushalt_neu

#haushalt1 = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_3_bearbeiten.xlsx')
#print(haushalt1)

timeindex = pd.date_range('00:00:00', '23:59:00', freq='1min')
#print(timeindex)
#haushalt = pd.Series(np.arange(1440), index=timeindex, dtype=float)
#print(haushalt)

#i = 0
#while i < 1440:
#    haushalt[i] = haushalt1[0][i]
#    i = i + 1
#print(haushalt)

#haushalt_neuspalte = pd.Series(np.arange(1440), index=timeindex, dtype=float)
#j=1440
#while j<525600+1:
#    print(j)
#    if j%1440 == 0:
#        haushalt = pd.concat([haushalt, haushalt_neuspalte], axis=1)
#        haushalt_neuspalte = pd.Series(np.arange(1440), index=timeindex, dtype=float)
#    if j == 525600: break
#    haushalt_neuspalte[j%1440] = haushalt1[0][j]
#    j = j + 1

#print(haushalt)
#haushalt.to_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_3_days.xlsx')

##haushalt1 = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_1_days.xlsx')
##haushalt2 = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_2_days.xlsx')
##haushalt3 = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_3_days.xlsx')
##haushalt = haushalt1.copy()
##i=0
##while i < 365:
##    haushalt = pd.concat([haushalt, haushalt2[i]], axis=1)
##    i = i +1

##i=0
##while i < 365:
##    haushalt = pd.concat([haushalt, haushalt3[i]], axis=1)
##    i = i +1

##print(haushalt)
##haushalt.to_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_123_days.xlsx')

haushalt = pd.read_excel('datasets/HTW Berlin/1min_htwBerlin/Haushalt_123_days_ohneCluster.xlsx')
print(haushalt)

###plt.figure(figsize=(16, 5))
###plt.xticks(fontsize=20)
###plt.yticks(fontsize=20)

###i = 0
###while i < 1095: #365*3
###    plt.plot(timeindex, haushalt[i],color='blue', alpha=0.12)
###    i=i+1

    # plt.plot(x, y, 's-',label=label)
###plt.xlabel('Zeit', fontsize=20)
###plt.ylabel('Wirkleistung[kW]', fontsize=20)
###plt.title('Wirkleistungen innerhalb von einem Tag', fontsize=20, ha='center')
###plt.savefig('datasets/Haushalte_123_inoneday.png', bbox_inches='tight')
###plt.show()

sillhoute_scores = []
n_cluster_list = np.arange(2,11).astype(int)

sc=MinMaxScaler()
haushalt=sc.fit_transform(haushalt)

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found=kmeans.fit_predict(haushalt)
    sillhoute_scores.append(silhouette_score(haushalt, kmeans.labels_))
    print('n_cluster: '+str(n_cluster)+' sillhoute: '+str(sillhoute_scores[n_cluster-2]))

plotten_blau(n_cluster_list, sillhoute_scores, title='Average sillhoute score per number of clusters',
                   xlabel='Number of clusters', ylabel='average sillhoute',datei='datasets/Sillhoute_inoneday_123.png')
