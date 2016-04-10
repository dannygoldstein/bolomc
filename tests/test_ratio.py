import os
import sncosmo
import numpy as np
import seaborn as sns
from bolomc import model
from matplotlib import pyplot as plt

sns.set_style('ticks')

def sever(s):
    splstr = s.split('/')
    dir = '/'.join(splstr[:-1])
    f = splstr[-1]
    return dir, f

pwd, _ = sever(__file__)

def path(snname):
    suffix = 'opt+nir_photo.dat'
    prefix = '../data/CSP_Photometry_DR2/%s'
    return os.path.join(pwd, prefix % (snname + suffix))

LC_W_NEG_RATIO = path('SN2005el')
LC_W_POS_RATIO = path('SN2009F')

magsys = sncosmo.get_magsystem('csp')

fc_pos = model.FitContext(LC_W_POS_RATIO)
fc_neg = model.FitContext(LC_W_NEG_RATIO)

def testNegRatio():
    assert any(fc_neg.lc['ratio'] < 0)

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

    for band in fc_neg.bands:
        sp = magsys.standards[band.name]
        binw = np.gradient(sp.wave)
        binterp = np.interp(sp.wave, band.wave, band.trans)
        prod = binterp * binw

        fig, ax = plt.subplots(figsize=(8,6))
        ax2 = ax.twinx()

        zeros = np.zeros_like(binterp)
        ax.fill_between(sp.wave, zeros, binterp, alpha=0.2)
        ax2.loglog(sp.wave, sp.flux, color='k')
        ax.set_xlabel('wavelength (AA)')
        ax.set_title(band.name)
        sns.despine()
        fig.savefig('output/%s.pdf' % band.name)
        del fig
