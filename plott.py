import pandas as pd # This is always assumed but is included here as an introduction.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import openpyxl
import xlwt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

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

def haushalt_einlesen():
    i=1
    haushalt=[0]*74
    while i<=74:
        print('Einlesen Haushalt'+str(i))
        haushalt[i-1] = indexieren_htw_xlsx('datasets/HTW Berlin/1min_htwBerlin/Haushalt_' +str(i)+'.csv' ,
                                'datasets/HTW Berlin/1min_htwBerlin/Haushalt_'+str(i)+'.xlsx')
        i=i+1
    return haushalt

def plotten_alle(haushalt):
    plt.figure(figsize=(16, 5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    i = 0
    while i < 74:
        plt.plot(haushalt[i].index, haushalt[i],color='blue', alpha=0.01)
        i=i+1

    # plt.plot(x, y, 's-',label=label)
    plt.xlabel('Zeit', fontsize=20)
    plt.ylabel('Wirkleistung[kW]', fontsize=20)
    plt.title('Wirkleistungen aller Haushalte', fontsize=20, ha='center')
    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig('datasets/alle_Haushalte.png', bbox_inches='tight')
    plt.show()

def concat_alle(haushalt):
    i = 1
    X = haushalt[0]
    while i < 74:
        X = pd.concat([X, haushalt[i]], axis=1)
        i=i+1
    return X

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

haushalt = haushalt_einlesen()
plotten_alle(haushalt)
X=concat_alle(haushalt)
print(X)
X.to_excel('datasets/HTW Berlin/1min_htwBerlin/X.xlsx')

sillhoute_scores = []
n_cluster_list = np.arange(2,31).astype(int)

sc=MinMaxScaler()
X=sc.fit_transform(X)

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found=kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    print('n_cluster: '+str(n_cluster)+' sillhoute: '+str(sillhoute_scores[n_cluster-2]))

plotten_blau(n_cluster_list, sillhoute_scores, title='Average sillhoute score per number of clusters',
                   xlabel='Number of clusters', ylabel='average sillhoute',datei='datasets/Sillhoute.png')