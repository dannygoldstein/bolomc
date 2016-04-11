import os
import sncosmo
import numpy as np
#import seaborn as sns
from bolomc import model
from matplotlib import pyplot as plt

from util_test import *

#sns.set_style('ticks')

SPARSE_LC = path('SN2005el')
fc = model.FitContext(SPARSE_LC)

def testWarpSparse():
    
    outdir = 'output/testWarpSparse'
    os.mkdir(outdir)
    
    fig, ax = plt.subplots(figsize=(5, 10))
    
    fc.one_iteration()
    sedw = fc.sedw.T
    res = ax.pcolorfast(fc.hsiao._wave, fc.hsiao._phase, sedw.T, cmap='RdBu',
                        vmin=-5, vmax=5)
    
    ax.plot(fc.lc['wave_eff'], fc.lc['mjd'], 'ro')
    ax.invert_yaxis()
    
    fig.colorbar(res)
    
    fig.savefig(os.path.join(outdir, 'sedwarp.pdf'))

    
    
    
