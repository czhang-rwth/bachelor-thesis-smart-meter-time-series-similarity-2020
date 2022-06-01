import pandas as pd # This is always assumed but is included here as an introduction.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal

def plot_kurven_glatt(t, y1, y2, y3,y4,title, xlabel, ylabel, label1, label2,label3,label4, png_datei, dpi=100):
    plt.figure(figsize=(16, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(t, y1,'o-', color='red', label=label1)
    plt.plot(t, y2,'s--', color='blue', label=label2)
    plt.plot(t, y3, 's--', color='green', label=label3)
    plt.plot(t, y4, 's--', color='orange', label=label4)
    #plt.plot(haushalt1.index, haushalt1, 'o-', color='red', label=label1)
    #plt.plot(haushalt2.index, haushalt2, 's--', color='blue', label=label2)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')
    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(png_datei, bbox_inches='tight')
    #plt.savefig('datasets/test.png', bbox_inches='tight')
    plt.show()

def plot_kurven_glatt_xcorr(t, y1, y2, y3,title, xlabel, ylabel, label1, label2,label3, png_datei, dpi=100):
    plt.figure(figsize=(16, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(t, y1,'s--', color='blue', label=label1)
    plt.plot(t, y2, 's--', color='green', label=label2)
    plt.plot(t, y3, 's--', color='orange', label=label3)
    #plt.plot(haushalt1.index, haushalt1, 'o-', color='red', label=label1)
    #plt.plot(haushalt2.index, haushalt2, 's--', color='blue', label=label2)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, ha='center')
    plt.legend(loc='best', fontsize=17)  # fuer label
    plt.savefig(png_datei, bbox_inches='tight')
    #plt.savefig('datasets/test.png', bbox_inches='tight')
    plt.show()

def xcorr(x,y):
    xcorr_wert = signal.correlate(x, y, mode='full', method='direct')
    return xcorr_wert

t = [1,2,3,4,5,6,7,8]

f = [8,9,11,7,4,6,5,10] #original
g = [11,12,14,10,7,9,8,13]
h = [5,7,8,6,2,5,2,8]
p = [18,17,16,15,14,13,12,11]

plot_kurven_glatt(t,f,g,h,p,title='Zeitreihe f(t), g(t), h(t) und p(t)',xlabel='Zeit t (min)', ylabel='Zeitreihenwert',
                       label1='f(t)',label2='g(t)',label3='h(t)',label4='p(t)',png_datei='datasets/Grundlage_Correlation.png')
print('f,g')
print(np.cov(f,g)[0,1])
print(np.corrcoef(f,g)[0,1])

print('f,h')
print(np.cov(f,h)[0,1])
print(np.corrcoef(f,h)[0,1])

print('f,p')
print(np.cov(f,p)[0,1])
print(np.corrcoef(f,p)[0,1])

xcorr_fg = xcorr(f,g)
xcorr_fh = xcorr(f,h)
xcorr_fp = xcorr(f,p)
xcorr_index = np.arange(1, len(xcorr_fg)+1, step=1)
plot_kurven_glatt_xcorr(xcorr_index, xcorr_fg, xcorr_fh, xcorr_fp, title='Kreuzkorrelation   f(t) x g(t)   f(t) x h(t)   f(t) x p(t)',
                        xlabel='Zeitverschiebung (min)',ylabel='Kreuzkorrelationswert', label1='f(t) x g(t)',label2='f(t) x h(t)',label3='f(t) x p(t)',
                        png_datei='datasets/Grundlage_Kreuzkorrelation.png')

print(xcorr_fg)
print(xcorr_fh)
print(xcorr_fp)

