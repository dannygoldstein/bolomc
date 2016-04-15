#!/usr/bin/env python 

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
from errors import *
from distributions import TruncNorm, stats
from vec import ParamVec

import emcee

######################################################
# CONSTANTS ##########################################
######################################################

h = 6.62606885e-27 # erg s
c = 2.99792458e10  # cm / s
AA_TO_CM = 1e-8 # dimensionless
ETA_SQ = 0.1
magsys = sncosmo.get_magsystem('csp')

######################################################

def filter_to_wave_eff(filt):
    filt = sncosmo.get_bandpass(filt)
    return filt.wave_eff

class FitContext(object):

    """Implementation of the PGM (Figure 1) from Goldstein & Kasen
    (2016). Defines a set of priors and a likelihood function for
    predicting bolometric light curves given broadband CSP photometry.

    """
    
    def __init__(self, lcfile, nphase, nlam,
                 dust_type=sncosmo.OD94Dust, 
                 exclude_bands=[],
                 rv_bintype='gmm',
                 outdir='output'):

        self.dust_type = dust_type
        self.exclude_bands = exclude_bands
        
        self.np = nphase
        self.nl = nlam
        self.outdir = outdir

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

        self.lp_prior = TruncNorm(0, np.inf, 3.5, 0.3)

        # This may be too tight
        self.llam_prior = TruncNorm(0, np.inf, 600., 15.)

        # set up coarse grid
        self.xstar_p = np.linspace(self.hsiao._phase[0], 
                                   self.hsiao._phase[-1], 
                                   self.np)
        self.xstar_l = np.linspace(self.hsiao._wave[0], 
                                   self.hsiao._wave[-1], 
                                   self.nl)
        
        # this is the coarse grid
        self.xstar = np.asarray(list(product(self.xstar_p, 
                                             self.xstar_l)))

        # get an initial guess for amplitude and t0
        
        guess_mod = sncosmo.Model(source=self.hsiao,
                                  effects=[self.dust_type(), sncosmo.F99Dust()],
                                  effect_names=['host', 'mw'],
                                  effect_frames=['rest', 'obs'])

        guess_mod.set(z=self.lc.meta['zcmb'])
        guess_mod.set(hostebv=self.ebv_prior.rvs())
        guess_mod.set(hostr_v=self.rv_prior.rvs())
        guess_mod.set(mwebv=self.mwebv)

        res, fitted_model = sncosmo.fit_lc(self.lc, guess_mod, ['amplitude','t0'])
        if not res['success']:
            raise FitError(res['message'])

        self.amplitude = res['parameters'][2]
        self.t0 = res['parameters'][1]
        self.hsiao_binw = np.gradient(self.hsiao._wave)
        

        self.lc['mjd'] = self.lc['mjd'] - self.t0
        self.obs_x = np.asarray(zip(self.lc['mjd'], self.lc['wave_eff']))
        self.rest_x = self.obs_x / (1 + self.lc.meta['zcmb'])

        # This is defined here and used repeatedly in the loglike
        # calculation.
        self.x = np.vstack((self.rest_x, self.xstar))
        self.diffmat = self.x[:, None] - self.x[None, :]
        

        

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
        #model.set(t0=self.t0)

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
        sqerr = stats.norm.logpdf(flux, 
                                  loc=self.lc['flux'], 
                                  scale=self.lc['fluxerr'])
        lp__ += np.sum(sqerr)

        # gaussian process likelihood calculation

        ratio = flux / self.lc['flux']
        ratio_m = ratio.mean()
        ratio_sd = ratio.std()
        ratio_scl = (ratio - ratio_m) / ratio_sd
        sedw_scl = (params.sedw - ratio_m) / ratio_sd
        y = np.concatenate((ratio_scl, sedw_scl.ravel()))

        l = np.asarray([params.lp, params.llam])
        sigma = self.diffmat / l
        sigma =  np.exp(-np.sum(sigma * sigma, axis=-1))
        sigma += np.diag(np.ones_like(y) * 1e-5)
        mu = np.zeros_like(y)
        
        lp__ += stats.multivariate_normal.logpdf(y, mean=mu, cov=sigma)

        return lp__

    def bolo(self, params):
        """Compute a bolometric flux curve for `params`."""
        flux = self._regrid_hsiao(params.sedw) * self.hsiao._passed_flux
        flux *= self.amplitude
        return np.sum(flux * self.hsiao_binw, axis=1)

    def __call__(self, params):
        try:
            vec = ParamVec(params, self.np, self.nl)
        except BoundsError as e:
            return -np.inf
        return self.logprior(vec) + self.loglike(vec)

    @property
    def D(self):
        return 4 + self.np * self.nl

# Define a helper function for the output formatting. 
stringify = lambda array: " ".join(["%.5e" % e for e in array])
    
def record(result, bfile, cfile, i):
    pos, lnprob, rstate = result
        
        # Dump chain state. 
        with open(cfile, 'a') as f:
            for k in range(pos.shape[0]):
                f.write("{3:4d} {0:4d} {2:f} {1:s}\n"
                        .format(k, stringify(pos[k]), lnprob[k], i))

        # Dump bolometric light curves.
        with open(bfile, 'a') as f:
            for k in range(pos.shape[0]):
                try:
                    vec = ParamVec(pos[k], fc.np, fc.nl)
                except BoundsError as e:
                    bolo = np.zeros(fc.hsiao._phase.shape[0]) * np.nan
                bolo = fc.bolo(vec)
                f.write("{0:4d} {1:4d} {2:s}\n"
                        .format(i, k, stringify(bolo)))
    

def main(lc_filename, nph, nl):
    
    # Fit a single light curve with the model.

    fc = FitContext(lc_filename, nph, nl)

    # create initial parameter vectors

    nwal = 2 * fc.D
    pvecs = list()

    diffs = fc.xstar[:, None] - fc.xstar[None, :]
    nmat = np.diag(np.ones(diffs.shape[0]) * 1e-5)
    
    for i in range(nwal):
        lp = fc.lp_prior.rvs()
        llam = fc.llam_prior.rvs()
        rv = fc.rv_prior.rvs()
        ebv = fc.ebv_prior.rvs()
        
        l = np.asarray([lp, llam])
        sigma = diffs / l
        sigma = ETA_SQ * np.exp(-np.sum(sigma * sigma, axis=-1))
        sigma += nmat
        mu = np.ones(sigma.shape[0])
        sedw = stats.multivariate_normal.rvs(mean=mu, cov=sigma)
        
        pvecs.append(np.concatenate(([lp, llam, rv, ebv], sedw)))
    
    # Set up the sampler. 
    sampler = emcee.EnsembleSampler(nwal, fc.D, fc)

    # Get set up to collect the output of the sampler. 
    
    if not os.path.exists(fc.outdir):
        os.mkdir(fc.outdir)
        
    chain_fname = os.path.join(fc.outdir, 'chain.dat')
    bolo_fname = os.path.join(fc.outdir, 'bolo.dat')

    chain_burn_fname = os.path.join(fc.outdir, 'chain_burn.dat')
    bolo_burn_fname = os.path.join(fc.outdir, 'bolo_burn.dat')

    # Clear the output files if they already exist. 
    with open(chain_fname, 'w') as f, open(bolo_fname, 'w'), \
         open(chain_burn_fname, 'w') as g, open(bolo_burn_fname, 'w'):
        f.write('np=%d nl=%d\n' % (fc.np, fc.nl))
        g.write('np=%d nl=%d\n' % (fc.np, fc.nl))


    # Do burn-in.
    sgen_burn = sampler.sample(pvecs, iterations=1000, storechain=False)

    for i, result in enumerate(sgen_burn):
        record(result, bolo_burn_fname, chain_burn_fname, i)
            
    # Set up sample generator.

    sgen = sampler.sample(pos, 
                          iterations=1000,
                          rstate0=state,
                          lnprob0=prob,
                          storechain=False)

    # Sample and record the output. 
    for i, result in enumerate(sgen):
        record(result, bolo_fname, chain_fname, i)
    


if __name__ == "__main__":

    lc_filename = sys.argv[1]
    nph = int(sys.argv[2])
    nl = int(sys.argv[3])
    main(lc_filename, nph, nl)
