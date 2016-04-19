#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

import os
import sys
import glob
import sncosmo
import numpy as np

from itertools import product
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize
from astropy.cosmology import Planck13
from astropy import units as u

from burns import *
from errors import *
from distributions import TruncNorm, stats
from vec import ParamVec

import h5py
import emcee

import logging

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
                 rv_bintype='gmm'):

        self.dust_type = dust_type
        self.exclude_bands = exclude_bands
        
        self.np = nphase
        self.nl = nlam

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

    def bolo(self, params, compute_luminosity=False):
        """Compute a bolometric flux curve for `params`."""
        flux = self._regrid_hsiao(params.sedw) * self.hsiao._passed_flux
        flux *= self.amplitude
        flux = np.sum(flux * self.hsiao_binw, axis=1)
        if compute_luminosity:
            distfac = 4 * np.pi * Planck13.luminosity_distance(self.lc.meta['zcmb']).to(u.cm).value**2
            lum = flux * distfac
            return lum
        return flux
        
    def Lfunc(self, params):
        x = self.hsiao._phase
        y = self.bolo(params, compute_luminosity=True)
        func = interp1d(x, y, kind='cubic')
        return func
        
    def tpeak(self, params, retfunc=False):
        func = self.Lfunc(params)
        
        def minfunc(t):
            # objective function
            return -func(t)
    
        res = minimize(minfunc, 0.)
        if not res.success:
            raise FitError(res.message)
        return res.x if not retfunc else (res.x, func)

    def Lpeak(self, params):
        tpeak, func = self.tpeak(params, retfunc=True)
        return func(tpeak)
    
    def dm15(self, params):
        tpeak, func = self.tpeak(params, retfunc=True)
        lpeak = func(tpeak)
        l15 = func(tpeak + 15)
        return 2.5 * np.log10(lpeak / l15)

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
    
def record(result, group, fc, i):
    pos, lnprob, rstate = result
    bolos = list()
    
    for k in range(pos.shape[0]):
        try:
            vec = ParamVec(pos[k], fc.np, fc.nl)
        except BoundsError as e:
            bolo = np.zeros(fc.hsiao._phase.shape[0]) * np.nan
        else:
            bolo = fc.bolo(vec, compute_luminosity=True)
        bolos.append(bolo)

    bolos = np.asarray(bolos)
    group['prob'][i] = lnprob
    group['bolo'][i] = bolos
    group['params'][i] = pos        

def rename_output_file(name):
    newname = name + '.old'
    if os.path.exists(newname):
        rename_output_file(newname)
    os.rename(name, newname)

def create_output_file(fname, fc):
    if os.path.exists(fname):
        rename_output_file(fname)
    f = h5py.File(fname)
    f['np'] = fc.np
    f['nl'] = fc.nl
    return f

def initialize_hdf5_group(group, fc, nsamp, nwal):
    group.create_dataset('bolo', (nsamp, nwal, fc.hsiao._phase.shape[0]), dtype='float64')
    group.create_dataset('params', (nsamp, nwal, fc.D), dtype='float64')
    group.create_dataset('prob', (nsamp, nwal), dtype='float64')
    return group
                         
def main(lc_filename, nph, nl, outfile, nburn=1000, nsamp=1000):
    
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
    # Rename the output files if they already exist. 
     
    out = create_output_file(outfile, fc)

    with out:
        out.create_dataset('init_params', data=pvecs)

        burn = out.create_group('burn')
        samp = out.create_group('samples')

        initialize_hdf5_group(burn, fc, nburn, nwal)
        initialize_hdf5_group(samp, fc, nsamp, nwal)

        # Do burn-in.
        sgen_burn = sampler.sample(pvecs, 
                                   iterations=nburn, 
                                   storechain=False)

        logging.info('beginning burn-in')
        for i, result in enumerate(sgen_burn):
            record(result, burn, fc, i)
            logging.info('burn-in iteration %d, med lnprob: %f',
                         i, np.median(result[1]))
        logging.info('burn-in complete')

        # Set up sample generator.
        pos, prob, state = result    
        sgen = sampler.sample(pos, 
                              iterations=nsamp,
                              rstate0=state,
                              lnprob0=prob,
                              storechain=False)

        logging.info('beginning sampling')
        # Sample and record the output. 
        for i, result in enumerate(sgen):
            record(result, samp, fc, i)
            logging.info('sampling iteration %d, med lnprob: %f',
                         i, np.median(result[1]))
        logging.info('sampling complete')
    


if __name__ == "__main__":

    lc_filename = sys.argv[1]
    nph = int(sys.argv[2])
    nl = int(sys.argv[3])
    outfile = sys.argv[4]
    logfile = sys.argv[5]
    logging.basicConfig(format='[%(asctime)s]: %(message)s',
                        filename=logfile, 
                        filemode='w',
                        level=logging.DEBUG)
    main(lc_filename, nph, nl, outfile)
