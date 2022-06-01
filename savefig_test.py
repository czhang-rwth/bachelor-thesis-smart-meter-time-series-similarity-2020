import pandas as pd # This is always assumed but is included here as an introduction.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

x = np.arange(1, 5, step=1)
y = np.arange(1, 5, step=1)

plt.figure(figsize=(16, 5))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.plot(x, y, label='wert')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('test of savefig', fontsize = 20, ha='center')
plt.legend(loc='best', fontsize = 17)
plt.savefig('datasets/test_savefig.png',bbox_inches = 'tight')
plt.show()