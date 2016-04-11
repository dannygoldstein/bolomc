import os
import sncosmo
import numpy as np
import seaborn as sns
from bolomc import model
from matplotlib import pyplot as plt

from util_test import *

sns.set_style('ticks')

LC_W_NEG_RATIO = path('SN2005el')
LC_W_POS_RATIO = path('SN2009F')

magsys = sncosmo.get_magsystem('csp')

fc_pos = model.FitContext(LC_W_POS_RATIO)
fc_neg = model.FitContext(LC_W_NEG_RATIO)

def testNegRatio():
    assert not any(fc_neg.lc['ratio'] < 0)

def testNeg():
    for band in fc_neg.bands:
        sp = magsys.standards[band.name]
        binterp = np.interp(sp.wave, band.wave, band.trans)
        binterp[binterp < 0] = 0
        assert all(binterp >= 0)

def testPosRatio():
    assert all(fc_pos.lc['ratio'] > 0)
    
def testInterp():

    magsys = sncosmo.get_magsystem('csp')

    outdir = 'output/testInterp'
    os.mkdir(outdir)

    for band in fc_neg.bands:
        sp = magsys.standards[band.name]
        binw = np.gradient(sp.wave)
        binterp = np.interp(sp.wave, band.wave, band.trans, left=0., right=0.)

        fig, ax = plt.subplots(figsize=(8,6))
        ax2 = ax.twinx()

        zeros = np.zeros_like(binterp)
        ax.fill_between(sp.wave, zeros, binterp, alpha=0.2)
        ax2.semilogx(sp.wave, sp.flux, color='k')
        ax.set_xlabel('wavelength (AA)')
        ax.set_title(band.name)

        ylim1 = ax.get_ylim()[1]
        ylim2 = ax2.get_ylim()[1]
        
        #ax.axhline(0, linestyle='--', color='k')

        #ax.set_ylim(-0.1, ylim1)
        #ax2.set_ylim(-0.1, ylim2)

        ax.axhline(y=0., linewidth=0.5, linestyle='--', color='k')
        
        ax.set_ylim(-0.1, 1.)
        ax2.set_ylim(-0.1 * ylim2, ylim2)

        fig.savefig(os.path.join(outdir, '%s.pdf' % band.name))
        del fig
