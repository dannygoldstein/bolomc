import os
import sncosmo
import numpy as np
#import seaborn as sns
from bolomc import model
from matplotlib import pyplot as plt

from numpy import testing

from util_test import *

#sns.set_style('ticks')

LC = path('SN2005el')
fc = model.FitContext(LC)

def testWarpSparse():
    
    outdir = 'output/testWarpSparse'
    os.mkdir(outdir)
    
    fig, ax = plt.subplots(figsize=(5, 10))
    
    fc.one_iteration()
    sedw = fc.sedw.T
    res = ax.pcolorfast(fc.hsiao._wave, fc.hsiao._phase, sedw.T, cmap='RdBu',
                        vmin=-2, vmax=2)
    
    ax.plot(fc.lc['wave_eff'], fc.lc['mjd'], 'ro')
    ax.invert_yaxis()
    
    fig.colorbar(res)
    
    fig.savefig(os.path.join(outdir, 'sedwarp.pdf'))

    
def testGPPhaseHyper():
    
    l_p = np.sqrt(1./fc.gp.theta_[0])

    testing.assert_array_less(l_p, 1e4)
    testing.assert_array_less(1, l_p)

def testGPWavelengthHyper():

    l_w = np.sqrt(1./fc.gp.theta_[1])
    
    testing.assert_array_less(l_w, 1e4)
    testing.assert_array_less(100, l_w)


def testPos():
    
    assert (fc.sedw >= 0).all()
