
import os
import sncosmo
import numpy as np
from bolomc import model
from matplotlib import pyplot as plt

from numpy import testing

from util_test import *

LC = path('SN2005el')
fc = model.FitContext(LC)

outdir = 'output/testColor'
os.mkdir(outdir)

def testColor():
    
    bolo = fc.one_iteration()
    flux = fc.hsiao._passed_flux * fc.sedw
    wave = fc.hsiao._wave
    phase = fc.hsiao._phase
    source = sncosmo.TimeSeriesSource(phase, wave, flux)
    model = fc._create_model(source=source)

    model.set(amplitude=fc.amplitude,
              t0=0.,
              z=fc.lc.meta['zcmb'])
    
    fig = sncosmo.plot_lc(data=fc.lc, model=model)
    fig.savefig(os.path.join(outdir, '2005el.pdf'))

    hsiao = sncosmo.Model(source='hsiao')
    hsiao.set(z=fc.lc.meta['zcmb'])
    res, fitted_model = sncosmo.fit_lc(data=fc.lc, model=model,
                                       vparam_names=['t0','amplitude'])
    
    fig = sncosmo.plot_lc(data=fc.lc, model=fitted_model)
    fig.savefig(os.path.join(outdir,'2005el_control.pdf'))
    
