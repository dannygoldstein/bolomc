import os
import sncosmo
import numpy as np
#import seaborn as sns
from bolomc import model

import matplotlib.cm as cm
from matplotlib import pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from numpy import testing

from util_test import *

#sns.set_style('ticks')

LC = path('SN2005el')
fc = model.FitContext(LC)

outdir = 'output/testWarp'
os.mkdir(outdir)


def testWarpSparse():
        
    fig, ax = plt.subplots(figsize=(5, 10))
    
    fc.one_iteration()
    sedw = fc.sedw.T
    
    base_cmap = cm.get_cmap('viridis')

    vmin = -1.
    vmax = 2.
    

    new_cmap = shiftedColorMap(base_cmap, start=0., midpoint=center(vmax,vmin),
                               stop=1.)
    res = ax.pcolorfast(fc.hsiao._wave, fc.hsiao._phase, sedw.T, cmap=new_cmap,
                        vmin=vmin, vmax=vmax)
    
    ax.plot(fc.lc['wave_eff'], fc.lc['mjd'], 'ro')
    ax.invert_yaxis()
    
    ax.set_xlabel('wavelength (AA)')
    ax.set_ylabel('phase (days)')
    
    fig.colorbar(res)
    
    fig.savefig(os.path.join(outdir, 'sedwarp.pdf'))

    
def testGPPhaseHyper():
    
    l_p = np.sqrt(1./fc.gp.theta_[0])

    testing.assert_array_less(l_p, 10)
    testing.assert_array_less(0.5, l_p)

def testGPWavelengthHyper():

    l_w = np.sqrt(1./fc.gp.theta_[1])
    
    testing.assert_array_less(l_w, 3000)
    testing.assert_array_less(100, l_w)


def testPos():
    
    assert (fc.sedw >= 0).all()

def testWarpAcc():

    gp_y = np.squeeze((fc.gp.y * fc.gp.y_std) + fc.gp.y_mean)
    gp_X = (fc.gp.X * fc.gp.X_std) + fc.gp.X_mean
    
    with PdfPages(os.path.join(outdir, 'testWarpAcc.pdf')) as pdf:
        
        for mjd, mjdgroup in fc.lc.to_pandas().groupby('mjd'):
            
            x_star = [(mjd, w) for w in fc.hsiao._wave]
            pred, mse = fc.gp.predict(x_star, eval_MSE=True)

            fig, ax = plt.subplots()
            ax.plot(fc.hsiao._wave, pred, 'k')
            ax.fill_between(fc.hsiao._wave, pred + np.sqrt(mse),
                            pred - np.sqrt(mse), color='b', alpha=0.5)
            
            ind = fc.obs_x[:, 0] == mjd
            
            #wave = gp_X[ind, 1]
            ratio = gp_y[ind]
            
            ax.plot(mjdgroup['wave_eff'], ratio, 'ro')
            ax.set_title(str(mjd))

            ax.set_xlabel('wavelength (AA)')
            ax.set_ylabel('ratio')
            
            pdf.savefig(fig)
            
            del fig
