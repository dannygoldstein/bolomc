#!/usr/bin/env python

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Template for test problems."

import numpy as np
import bolomc 
import sncosmo
from astropy.table import Table

csp = sncosmo.get_magsystem('csp')
hsiao = sncosmo.get_source('hsiao', version='3.0')
gain = 50000.


class Problem(object):

    def __init__(self, sedw, rv, ebv, lp, llam, # "True parameter values."
                 z, mwebv, # "True parameter values."
                 dust_type=sncosmo.OD94Dust, 
                 template=hsiao):

        self.template = template
        p = self.template._phase
        l = self.template._wave
        flux = self.template.flux(p, l)

        try:
            sedw.__call__
        except AttributeError:
            wflux = sedw * flux
        else:
            wsurf = np.asarray([[sedw(pp, ll) for ll in l] for pp in p])
            wflux = wsurf * flux 
        self.source = sncosmo.TimeSeriesSource(p, l, wflux)
        
        self.rv = rv
        self.ebv = ebv
        self.lp = lp
        self.llam = llam
        self.z = z
        self.dust_type = dust_type
        self.mwebv = mwebv
        
    @property
    def model(self):
        model = sncosmo.Model(source=self.source, 
                              effects=[self.dust_type(), sncosmo.F99Dust()],
                              effect_names=['host','mw'],
                              effect_frames=['rest','obs'])
        
        model.set(z=self.z)
        model.set(mwebv=self.mwebv)
        model.set(hostr_v=self.rv)
        model.set(hostebv=self.ebv)
        model.set(t0=0.)
        model.set_source_peakabsmag(-19.0, 'bessellb', 'ab')
        return model

    def data(self, nobs, exclude_bands=[]):
        model = self.model
        baset = np.linspace(model.mintime(), model.maxtime(), nobs)
        zps = []
        zpsyss = []
        times = []
        bands = []
        skynoises = []
        for band in map(sncosmo.get_bandpass, csp.bands):
            if band.name in exclude_bands:
                continue
            zps += ([csp.zeropoints[band.name]] * nobs)
            zpsyss += ([csp.name] * nobs)
            times += baset.tolist()
            skynoises += (1e-10 * model.bandflux(band, baset)).tolist()
            bands += [band.name] * nobs
        gains = np.ones_like(times) * gain
        obs = Table({'time':times,
                     'band':bands,
                     'gain':gains,
                     'skynoise':skynoises,
                     'zp':zps,
                     'zpsys':zpsyss})
        params = [dict(zip(model.param_names, model.parameters))]
        lcs = sncosmo.realize_lcs(obs, model, params)
        lc = sncosmo.photdata.normalize_data(lcs[0])
        lc['fluxerr'] = 0.01 * lc['flux'].max()
        return lc
