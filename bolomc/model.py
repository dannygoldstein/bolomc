
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

import os
import sys
import sncosmo
import numpy as np

from itertools import product
from scipy.interpolate import RectBivariateSpline

from burns import *
from exceptions import *
from distributions import TruncNorm, stats

######################################################
# CONSTANTS ##########################################
######################################################

h = 6.62606885e-27 # erg s
c = 2.99792458e10  # cm / s
AA_TO_CM = 1e-8 # dimensionless
magsys = sncosmo.get_magsystem('csp')

######################################################

def filter_to_wave_eff(filt):
    filt = sncosmo.get_bandpass(filt)
    return filt.wave_eff

def compute_ratio(band, type='flux'):

    """Compute 
   
        S alpha(\lambda) T(\lambda) d\lambda 
        ------------------------------------
              S T(\lambda) d\lambda
   
    if `type` == 'flux', or

        S \lambda \alpha(\lambda) T(\lambda) / (hc) d\lambda
        ----------------------------------------------------
                   S T(\lambda) d\lambda
    
    if `type` == 'photon',

    where T and alpha are the transmission function and standard
    spectrum of `band`, respectively."""

    band = sncosmo.get_bandpass(band)
    sp = magsys.standards[band.name] # flux-calibrated standard spectrum
    binw = np.gradient(sp.wave) # should be 10AA

    # interpolate the bandpass wavelength grid to the spectrum
    # wavelength grid, setting the transmission to 0. at wavelengths
    # that are beyond the grid defining the bandpass

    binterp = np.interp(sp.wave, band.wave, band.trans, left=0., right=0.)

    # ensure all transmissions are positive
    binterp[binterp < 0] = 0
    
    # compute the product alpha(\lambda) d\lambda    
    prod = binterp * binw
    
    # do the first integral 
    if type == 'flux':
        numerator = np.sum(sp.flux * prod) 
    elif type == 'photon':
        numerator = np.sum(sp.flux * sp.wave * prod) / (h * c) * AA_TO_CM
    
    # do the second integral
    denomenator = np.sum(prod)
    
    return numerator / denomenator


class FitContext(object):

    """Implementation of the PGM (Figure 1) from Goldstein & Kasen
    (2016). Defines a set of priors and a likelihood function for
    predicting bolometric light curves given broadband CSP photometry.

    """
    
    def __init__(self, lcfile, np, nl, dust_type=sncosmo.OD94Dust, exclude_bands=[],
                 rv_bintype='gmm'):

        self.dust_type = dust_type
        self.exclude_bands = exclude_bands
        
        self.np = np
        self.nl = nl

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
        
        self.ebv_prior = TruncNorm(0., np.inf, self.host_ebv, 
                                   self.host_ebv_err)

        self.rv_prior  = get_hostrv_prior(self.lc.meta['name'],
                                          self.rv_bintype,
                                          self.dust_type)

        # set up coarse grid
        self.xstar_p = np.linspace(self.hsiao._phase[0], 
                                   self.hsiao._phase[1], 
                                   self.np)
        self.xstar_l = np.linspace(self.hsiao._wave[0], 
                                   self.hsiao._wave[1], 
                                   self.nl)
        
        # this is the coarse grid
        self.xstar = np.asarray(list(product(self.xstar_p, 
                                             self.xstar_l)))

        # get an initial guess for amplitude and t0
        
        empty_arr = np.zeros(4 + self.np * self.nl)
        guess_vec = ParamVec(empty_arr, self.np, self.nl)

        guess_vec.ebv = self.ebv_prior.rvs()
        guess_vec.rv = self.rv_prior.rvs()
        
        # harmless hack to avoid bounds errors
        guess_vec.lt = 0.1
        guess_vec.llam = 0.1

        guess_mod = self._create_model(guess_vec)
        
        res, fitted_model = sncosmo.fit_lc(self.lc, guess_mod, ['amplitude','t0'])
        if not res['success']:
            raise FitError(res['message'])

        # TODO:: check to see if this is still right
        self.amplitude = res['parameters'][2]
        self.t0 = res['parameters'][1]

        # Deprecated

        '''
        self.lc['mag'] = -2.5 * np.log10(self.lc['flux']) + self.lc['zp']
        self.lc['ms'] = map(magsys.standard_mag, self.lc['filter'])
        ''' 
        
        self.hsiao_binw = np.gradient(self.hsiao._wave)
        
        # Deprecated 

        '''
        # compute the band ratio once for each band 
        self.ratio = {band.name : compute_ratio(band, type='photon') \
                      for band in self.bands}

        self.lc['ratio'] = [self.ratio[filt] for filt in self.lc['filter']]
        '''

        self.lc['mjd'] = self.lc['mjd'] - self.t0
        self.obs_x = np.asarray(zip(self.lc['mjd'], self.lc['wave_eff']))
        self.rest_x = self.obs_x / (1 + self.lc.meta['zcmb'])

        # This is defined here and used repeatedly in the loglike
        # calculation.
        self.x = np.vstack((self.rest_x, self.xstar))

        # Deprecated 

        '''
        # quantities you only need to compute once
        self.numer = 10**(0.4 * (self.lc['ms'] - self.lc['mag'])) * \
                     self.lc['ratio'] * (1 + self.lc.meta['zcmb'])
        
        self.S_R = np.asarray([self.hsiao.flux(*tup) * \
                               self.amplitude * tup[1] / (h * c) * AA_TO_CM \
                               for tup in self.rest_x]) # monochromatic
                                                        # photon flux

        '''

        

    def _regrid_hsiao(self, warp_f):
        """Take an SED warp matrix defined on a coarse grid and interpolate it
        to the hsiao grid using a cubic spline.

        """

        spl = RectBivariateSpline(self.xstar_p, 
                                  self.xstar_l,
                                  warp_f)
        
        return spl(self.hsiao._phase,
                   self.hsiao._wave)
        
                                                        
    def _create_model(self, params):
        """If source is None, use Hsiao."""
        
        # warp the SED
        flux = self._regrid_hsiao(params.sedw) * self.hsiao._passed_flux
        source = sncosmo.TimeSeriesSource(self.hsiao._phase,
                                          self.hsiao._wave,
                                          flux)
        
        model = sncosmo.Model(source=source,
                              effects=[self.dust_type(), sncosmo.F99Dust()],
                              effect_names=['host','mw'],
                              effect_frames=['rest','obs'])

        # spectroscopic redshift
        model.set(z=self.lc.meta['zcmb'])
        
        # MW Rv is fixed at 3.1.  mwebv could be a parameter but we
        # will fix it for simplicity
        model.set(mwebv=self.mwebv)

        # Draw random host reddening parameters. 
        model.set(hostr_v=params.rv)
        model.set(hostebv=params.ebv)
        
        # set source parameters
        # these are fixed
        
        model.set(amplitude=self.amplitude)
        model.set(t0=self.t0)

        return model

    def logprior(self, params):
        """Compute logp(params)."""

        lp__ = 0
        
        lp__ += self.ebv_prior(params.ebv)
        lp__ += self.rv_prior(params.rv)

        # do gaussian priors around the guesses for A and t0
        
        return lp__

    def loglike(self, params):
        """Compute loglike(params)."""
        
        lp__ = 0
        model = self._create_model(params)
        flux = model.bandflux(self.lc['filter'], 
                              self.lc['mjd'])

        # model / data likelihood calculation 
        sqerr = ((flux - self.lc['flux']) / self.lc['fluxerr'])**2
        lp__ += np.sum(sqerr)

        # gaussian process likelihood calculation

        ratio = flux / self.lc['flux']
        ratio_m = ratio.mean()
        ratio_sd = ratio.std()
        ratio_scl = (ratio - ratio_m) / ratio_sd
        sedw_scl = (params.sedw - ratio_m) / ratio_sd
        y = np.concatenate((ratio_scl, sedw_scl.ravel()))

        l = np.asarray([params.lp, params.llam])
        sigma = (self.x[:, None] - self.x[None, :]) / l
        sigma = np.exp(-np.sum(sigma * sigma), axis=-1)
        sigma += np.diag(np.ones_like(y) * 1e-5)
        mu = np.zeros_like(y)
        
        lp__ += stats.multivariate_normal.logpdf(y, mean=mu, cov=sigma)

        return lp__

    def bolo(self, params):
        """Compute a bolometric flux curve for `params`."""
        flux = self._regrid_hsiao(params.sedw) * self.hsiao._passed_flux
        flux *= params.amplitude
        return np.sum(flux * self.hsiao_binw, axis=1)

    def __call__(self, params):
        try:
            vec = ParamVec(params, self.np, self.nl):
        except BoundsError as e:
            return -np.inf
        return self.logprior(vec) + self.loglike(vec)
    
    

if __name__ == "__main__":
    
    # Fit a single light curve with the model.

    lc_filename = sys.argv[1]
    fc = FitContext(lc_filename)
    
    # Run the model. 
    lcs = np.asarray([fc.one_iteration() for i in range(500)])
