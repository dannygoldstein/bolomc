import numpy as np
from astropy.io.ascii import read
from scipy.stats import spearmanr

distmod = read('J_AJ_139_120/table1.dat', readme='ReadMe')
lcparam = read('J_AJ_139_120/table2.dat', readme='ReadMe')
lcparam = lcparam[lcparam['Band'] == 'B']

# filter away sne that are not "best"
distmod = distmod[distmod['Best'] == 'Yes']
dm15B = distmod['Dm15(B)']
mu = distmod['<mu>']
mmax = np.asarray([lcparam[lcparam['SN'] == n]['mmax'][0] for n in distmod['SN']])
Mmax = mmax - mu
print spearmanr(dm15B, Mmax)
