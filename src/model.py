
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

import os
import sys
import sncosmo
import numpy as np

from itertools import product
from sklearn.gaussian_process import GaussianProcess

from burns import *
from exceptions import *
from distributions import TruncNorm

def filter_to_wave_eff(filt):
    filt = sncosmo.get_bandpass(filt)
    return filt.wave_eff

#        t,  lambda
TH_LO = [1e-2, 1e-2]
TH_HI = [1.e1, 1.e1]
TH_0  = [1e-1, 1e-1]

NUG = 1e-5

class FitContext(object):

    """Implementation of the PGM (Figure 1) from Goldstein & Kasen
    (2016). Defines a set of priors and a likelihood function for
    predicting bolometric light curves given broadband CSP photometry.
    """
    
    def __init__(self, lcfile, dust_type=sncosmo.OD94Dust, exclude_bands=[],
                 rv_bintype='gmm'):

        self.dust_type = dust_type
        self.exclude_bands = exclude_bands
        self.lc = sncosmo.read_lc(lcfile, format='csp')
        self.lc['wave_eff'] = map(filter_to_wave_eff, self.lc['filter'])
        self.lc.sort(['mjd', 'wave_eff'])
        self.mwebv, _ = get_mwebv(self.lc.meta['name'])

        self.rv_bintype = rv_bintype
        
        self.host_ebv, self.host_ebv_err = get_hostebv(self.lc.meta['name'])

        self.bands = [sncosmo.get_bandpass(band) for
                      band in np.unique(self.lc['filter']) if
                      band not in self.exclude_bands]

        # Load Hsiao SED into memory here so you don't have to load it
        # every time create_model is called.

        self.hsiao = sncosmo.get_source('hsiao', version='3.0')
        
        # Set up priors.
        
        self.ebv_dist = TruncNorm(0., np.inf, self.host_ebv, self.host_ebv_err)
        self.rv_dist  = get_hostrv_prior(self.lc.meta['name'],
                                         self.rv_bintype,
                                         self.dust_type)

        # fit amplitude and t0 
        model = self._create_model()
        res, fitted_model = sncosmo.fit_lc(self.lc, model, ['amplitude','t0'])
        if not res['success']:
            raise FitError(res['message'])

        self.amplitude = res['parameters'][2]
        self.t0 = res['parameters'][1]
        self.gp = GaussianProcess(thetaL=TH_LO, 
                                  thetaU=TH_HI, 
                                  theta0=TH_0, 
                                  nugget=NUG)

        self.lc['mag'] = -2.5 * np.log10(self.lc['flux']) + self.lc['zp']
        magsys = sncosmo.get_magsystem('csp')
        self.lc['ms'] = [magsys.standard_mag(band) for band in self.lc['filter']]
        
        self.hsiao_binw = np.gradient(self.hsiao._wave)
        
        self.ratio = {}
        for band in self.bands:
            sp = magsys.standards[band.name]
            binw = np.gradient(sp.wave)
            binterp = np.interp(sp.wave, band.wave, band.trans)
            prod = binterp * binw
            self.ratio[band.name] = np.sum(sp.flux * prod) / np.sum(prod)

        self.lc['ratio'] = [self.ratio[filt] for filt in self.lc['filter']]

        ### re-examine this
        self.lc['mjd'] = self.lc['mjd'] - self.t0

        self.obs_x = np.asarray(zip(self.lc['mjd'], self.lc['wave_eff']))
        self.rest_x = self.obs_x / (1 + self.lc.meta['zcmb'])

        self.gp_xstar = np.asarray(list(product(self.hsiao._phase,
                                                self.hsiao._wave)))


        # quantities you only need to compute once
        self.numer = 10**(0.4 * (self.lc['ms'] - self.lc['mag'])) * \
                     self.lc['ratio'] * (1 + self.lc.meta['zcmb'])
        
        self.S_R = np.asarray([self.hsiao.flux(*tup) * self.amplitude for tup in self.rest_x])


    def _create_model(self, source=None):
        """If source is None, use Hsiao."""
    
        if source is None:
            source = self.hsiao
        
        model = sncosmo.Model(source=source,
                              effects=[self.dust_type(), sncosmo.F99Dust()],
                              effect_names=['host','mw'],
                              effect_frames=['rest','obs'])

        # spectroscopic redshift
        model.set(z=self.lc.meta['zcmb'])
        
        # MW Rv is fixed at 3.1. 
        model.set(mwebv=self.mwebv)

        # Draw random host reddening parameters. 
        model.set(hostr_v=self.rv_dist.rvs())
        model.set(hostebv=self.ebv_dist.rvs())
        
        try:
            model.set(amplitude=self.amplitude)
        except AttributeError:
            pass

        try:
            model.set(t0=self.t0)
        except AttributeError:
            pass

        return model

        
    def mw_dust(self):
        dust = sncosmo.F99Dust()
        dust.set(ebv=self.mwebv)
        return dust

    def host_dust(self):
        dust = self.dust_type()
        dust.set(ebv=self.ebv_dist.rvs())
        dust.set(r_v=self.rv_dist.rvs())
        return dust

    def one_iteration(self):
        
        # Create dust
        host_dust = self.host_dust()
        mw_dust = self.mw_dust()
    
        # Determine the warping at the knot points
        D_R = host_dust.propagate(self.rest_x[:, 1], 1.) # second
                                                         # argument is
                                                         # a hack to
                                                         # just get
                                                         # the
                                                         # transmission
        D_O = mw_dust.propagate(self.obs_x[:, 1], 1.)    # ditto
        
        denom = D_R * self.S_R * D_O
        
        # try doing (numer / denom) - 1
        warp = self.numer / denom

        # train GP
        self.gp.fit(self.rest_x, warp)
    
        # interpolate
        pf = self.hsiao._passed_flux
        sedw = self.gp.predict(self.gp_xstar).reshape(pf.shape)
        flux = self.amplitude * sedw * pf 
        bolo = np.sum(flux * self.hsiao_binw, axis=1)
        return bolo

if __name__ == "__main__":
    
    # Fit a single light curve with the model.

    lc_filename = sys.argv[1]
    fc = FitContext(lc_filename)
    
    # Run the model. 
    lcs = np.asarray([fc.one_iteration() for i in range(500)])
