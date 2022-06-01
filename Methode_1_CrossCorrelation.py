import pandas as pd # This is always assumed but is included here as an introduction.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal

def indexieren_htw_xlsx(csv_Datei ,excel_Datei):
    haushalt_csv = pd.read_csv(csv_Datei)
    print(haushalt_csv)
    haushalt_csv = haushalt_csv['Timestamp;PL1[W];PL2[W];PL3[W];QL1[W];QL2[W];QL3[W]'].str.split(';', expand=True)
    print(haushalt_csv)
    #######################haushalt_csv.to_excel(excel_Datei)
    haushalt = pd.read_excel(excel_Datei)
    len_0 = len(haushalt)
    print(haushalt)

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
    print(haushalt)
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

def plotten(x, y, title='', xlabel='', ylabel='', label='', dpi=100):
    plt.figure(figsize=(16, 5))
    plt.plot(x, y, 's-',label=label)
    #plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def plotten_glatt(x, y, title, xlabel, ylabel, label, png_datei, dpi=100):
    plt.figure(figsize=(16, 5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.plot(x, y, 's-',label=label)
    plt.plot(x, y, label=label)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')

    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(png_datei, bbox_inches='tight')
    plt.show()

def plotten_glatt_green(x, y, title, xlabel, ylabel, label, png_datei, dpi=100):
    plt.figure(figsize=(16, 5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.plot(x, y, 's-',label=label)
    plt.plot(x, y, color='green', label=label)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')

    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(png_datei, bbox_inches='tight')
    plt.show()

def plot_zwei_kurven_glatt(x1, y1, x2, y2, title, xlabel, ylabel, label1, label2, png_datei, dpi=100):
    plt.figure(figsize=(16, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(x1, y1, color='blue', label=label1)
    plt.plot(x2, y2, color='red', label=label2)
    #plt.plot(haushalt1.index, haushalt1, 'o-', color='red', label=label1)
    #plt.plot(haushalt2.index, haushalt2, 's--', color='blue', label=label2)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')
    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(png_datei, bbox_inches='tight')
    #plt.savefig('datasets/test.png', bbox_inches='tight')
    plt.show()

def xkorrelation(haushalt1, haushalt2, title=''):
    # hier gilt len(haushalt1)=len(haushalt2)
    i = 1  # wie viele Punkte insgesamt
    xcorr_index = 0

    xcorr = [0] * 500
    while i <= 500:
        j = 1
        sum = 0
        while j <= i:
            sum = sum + haushalt1.head(i)[j-1] * haushalt2.tail(i)[j-1]
            j=j+1
        xcorr[xcorr_index] = sum.copy()
        print(i, xcorr[xcorr_index])
        xcorr_index = xcorr_index + 1
        i = i + 1

    achse_x = np.arange(1, 501, step=1)
    achse_x_func = np.arange(1,501)

    A_1, B_1 = optimize.curve_fit(f_1, achse_x, xcorr)[0]
    print(A_1, B_1)
    xcorr_1 = A_1 *achse_x_func+B_1

    A_2, B_2, C_2 = optimize.curve_fit(f_2, achse_x, xcorr)[0]
    print(A_2, B_2, C_2)
    xcorr_2 = A_2*pow(achse_x_func+B_2,2)+C_2

    A_3, B_3, C_3 = optimize.curve_fit(f_3, achse_x, xcorr)[0]
    print(A_3, B_3, C_3)
    xcorr_3 = A_3 * pow(achse_x_func + B_3, 3) + C_3

    A_4, B_4, C_4 = optimize.curve_fit(f_4, achse_x, xcorr)[0]
    print(A_4, B_4, C_4)
    xcorr_4 = A_4 * pow(achse_x_func + B_4, 4) + C_4

    #A_5, B_5, C_5 = optimize.curve_fit(f_5, achse_x, xcorr)[0]
    #print(A_5, B_5, C_5)
    #xcorr_5 = A_5 * pow(achse_x_func + B_5, 5) + C_5

    #A_n, B_n, C_n, D_n = optimize.curve_fit(f_n, achse_x, xcorr)[0]
    #print(A_n, B_n, C_n, D_n)
    #xcorr_n = A_n * pow(achse_x_func + B_n, D_n) + C_n

    plotten_glatt(achse_x,xcorr, title=title, xlabel='Zeitverschiebung in Minuten', ylabel='Kreuzkorrelation',
                  label='Kreuzkorrelationskurve',png_datei='datasets/Kreuzkorrelation_Haushalt_1&2.png', dpi=100)

    plot_zwei_kurven_glatt(achse_x,xcorr,achse_x_func,xcorr_1,title=title,xlabel='Zeitverschiebung in Minuten',
                           ylabel='Kreuzkorrelation',label1='Kreuzkorrelationskurve',label2='Lineare Anpassung der Kreuzkorrelationskurve',
                           png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_linear.png',dpi=100)

    plot_zwei_kurven_glatt(achse_x, xcorr, achse_x_func, xcorr_2, title=title, xlabel='Zeitverschiebung in Minuten',
                           ylabel='Kreuzkorrelation', label1='Kreuzkorrelationskurve',
                           label2='Quadratische Anpassung der Kreuzkorrelationskurve',
                           png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_quadratisch.png', dpi=100)

    plot_zwei_kurven_glatt(achse_x, xcorr, achse_x_func, xcorr_3, title=title, xlabel='Zeitverschiebung in Minuten',
                           ylabel='Kreuzkorrelation', label1='Kreuzkorrelationskurve',
                           label2='Anpassung 3ter Potenz der Kreuzkorrelationskurve',
                           png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_3.png', dpi=100)

    plot_zwei_kurven_glatt(achse_x, xcorr, achse_x_func, xcorr_4, title=title, xlabel='Zeitverschiebung in Minuten',
                           ylabel='Kreuzkorrelation', label1='Kreuzkorrelationskurve',
                           label2='Anpassung 4ter Potenz der Kreuzkorrelationskurve',
                           png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_4.png', dpi=100)

    #plot_zwei_kurven_glatt(achse_x, xcorr, achse_x_func, xcorr_5, title=title, xlabel='Zeitverschiebung in Minuten',
    #                       ylabel='Kreuzkorrelation', label1='Kreuzkorrelationskurve',
    #                       label2='5 Anpassung der Kreuzkorrelationskurve',
    #                       png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_5.png', dpi=100)

    #plot_zwei_kurven_glatt(achse_x, xcorr, achse_x_func, xcorr_n, title=title, xlabel='Zeitverschiebung in Minuten',
    #                       ylabel='Kreuzkorrelation', label1='Kreuzkorrelationskurve',
    #                       label2='n Anpassung der Kreuzkorrelationskurve',
    #                       png_datei='datasets/Kreuzkorrelation_Haushalt_1&2_n.png', dpi=100)


def f_1(x,A,B):
    return A*x+B

def f_2(x,A,B,C):
    return A*pow(x+B,2)+C

def f_3(x,A,B,C):
    return A*pow(x+B,3)+C

def f_4(x,A,B,C):
    return A*pow(x+B,4)+C

def f_5(x,A,B,C):
    return A*pow(x+B,5)+C

def f_n(x,A=float,B=float,C=float,D=int):
    return A*pow(x+B,D)+C

def f_5(x,A,B,C,D,E,F):
    return A*x+B*pow(x,2)+C*pow(x,3)+D*pow(x,4)+E*pow(x,5)+F

def f_log(x,A,B,C,D):
    return A*np.log(B*x+C)+D

def manhattan_distance(x, y): #, title=''):  #x und y müssen gleiche zeitreihe haben
    i=0
    manhattan=[0]*len(x)

    while i<len(x):
        manhattan[i] = abs(x[i]-y[i])
        i=i+1

    return manhattan

print("Einlese Haushalt 1")
haushalt1 = indexieren_htw_xlsx('datasets/HTW Berlin/1min_htwBerlin/Haushalt_1.csv' ,
                                'datasets/HTW Berlin/1min_htwBerlin/Haushalt_1.xlsx')

plotten_glatt(haushalt1.index, haushalt1, title='Wirkleistungen Haushalt 1', xlabel='Zeit', ylabel='Wirkleistung P[kW]',
              label='Wirkleistung Haushalt1',png_datei='datasets/Wirkleistung_Haushalt1.png',dpi=100)

print("Einlese Haushalt 2")
haushalt2 = indexieren_htw_xlsx('datasets/HTW Berlin/1min_htwBerlin/Haushalt_2.csv',
                                'datasets/HTW Berlin/1min_htwBerlin/Haushalt_2.xlsx')

plotten_glatt_green(haushalt2.index, haushalt2, title='Wirkleistungen Haushalt 2', xlabel='Zeit', ylabel='Wirkleistung P[kW]',
              label='Wirkleistung Haushalt2',png_datei='datasets/Wirkleistung_Haushalt2.png',dpi=100)

xkorrelation(haushalt1,haushalt2,title='Kreuzkorrelation von Wirkleistungen Haushalt1 und 2')