#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

__all__ = ['FitContext', 'main']

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
import argparse

######################################################
# CONSTANTS ##########################################
######################################################

h = 6.62606885e-27 # erg s
c = 2.99792458e10  # cm / s
AA_TO_CM = 1e-8 # dimensionless
ETA_SQ = 0.1
NUG = 1e-7 
magsys = sncosmo.get_magsystem('csp')

######################################################
# DEFAULTS ###########################################
######################################################

LOGFILE = None # Use stdout.
NBURN = 1000 # Number of burn-in iterations.
NSAMP = 1000 # Number of sampling iterations.
NL = None # Number of wavelength knots (regular grid).
NWALKERS = 200 # Number of walkers in the ensemble.
NTHREADS = 1 # Number of threads for MCMC sampling. 
EXCLUDE_BANDS = [] # Fit all bandpasses given. 
DUST_TYPE = 'OD94' # Host galaxy dust reddening law.
RV_BINTYPE = 'gmm' # Host galaxy Rv prior type. 
SPLINT_ORDER = 3 # Spline interpolation order.

######################################################
# HELPERS ############################################
######################################################

def filter_to_wave_eff(filt):
    filt = sncosmo.get_bandpass(filt)
    return filt.wave_eff
    
def record(result, group, fc, sampler, i):
    pos, lnprob, rstate = result
    bolos = list()
    
    for k in range(pos.shape[0]):
        try:
            vec = ParamVec(pos[k], fc.nph, fc.nl)
        except BoundsError as e:
            bolo = np.zeros(fc.hsiao._phase.shape[0]) * np.nan
        else:
            bolo = fc.bolo(vec, compute_luminosity=True)
        bolos.append(bolo)

    bolos = np.asarray(bolos)
    group['prob'][i] = lnprob
    group['bolo'][i] = bolos
    group['params'][i] = pos        
    group['afrac'][i] = sampler.acceptance_fraction

def rename_output_file(name):
    newname = name + '.old'
    if os.path.exists(newname):
        rename_output_file(newname)
    os.rename(name, newname)

def create_output_file(fname, fc):
    if os.path.exists(fname):
        rename_output_file(fname)
    f = h5py.File(fname)
    f['nph'] = fc.nph
    f['nl'] = fc.nl
    return f

def initialize_hdf5_group(group, fc, nsamp, nwalkers):

    bs = (nsamp, nwalkers, fc.hsiao._phase.shape[0])
    afs = (nsamp, nwalkers)
    pars = (nsamp, nwalkers, fc.D)
    probs = (nsamp, nwalkers)

    group.create_dataset('params', pars, dtype='float64')
    group.create_dataset('prob', probs, dtype='float64')
    group.create_dataset('afrac', afs, dtype='float64')
    group.create_dataset('bolo', bs, dtype='float64')

    return group

def dust(s):
    """Convert the string representation of `s` of an sncosmo dust class
    to its class."""
    if s == 'OD94':
        return sncosmo.OD94Dust
    elif s == 'F99':
        return sncosmo.F99Dust
    else:
        raise ValueError('Invalid dust type.')

######################################################

class FitContext(object):

    """Implementation of the PGM (Figure 1) from Goldstein & Kasen
    (2016). Defines a set of priors and a likelihood function for
    predicting bolometric light curves given broadband CSP photometry.

    """
    
    def __init__(self, lc_filename, nph, nl=NL, dust_type=DUST_TYPE,
                 exclude_bands=EXCLUDE_BANDS, rv_bintype=RV_BINTYPE,
                 splint_order=SPLINT_ORDER):

        self.dust_type = dust(dust_type)
        self.exclude_bands = exclude_bands
        self.splint_order = splint_order
        
        self.nph = nph
        self.nl = nl

        self.lc = sncosmo.read_lc(lc_filename, format='csp')
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
                                   self.nph)

        # If no regular wavelength grid is specified...
        if self.nl is None:
            #...make the wavelength grid the effective wavelengths of
            #the filters.
            self.xstar_l = np.array(sorted(map(filter_to_wave_eff, self.bands)))
            self.nl = self.xstar_l.size
        else:
            #...else lay down a regular grid with `nl` points over the
            #Hsiao domain.
            self.xstar_l = np.linspace(self.hsiao._wave[0],
                                       self.hsiao._wave[-1],
                                       self.nl)
        
        # Weave the full grid [a list of (phase, wavelength) points]. 
        self.xstar = np.asarray(list(product(self.xstar_p, 
                                             self.xstar_l)))

        # Get an initial guess for amplitude and t0.
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

        # This is defined here and used repeatedly in the logprior
        # calculation.
    
        self.diffmat = self.xstar[:, None] - self.xstar[None, :]
        

    def _regrid_hsiao(self, warp_f):
        """Take an SED warp matrix defined on a coarse grid and interpolate it
        to the hsiao grid using a spline.

        """
        
        spl = RectBivariateSpline(self.xstar_p, 
                                  self.xstar_l,
                                  warp_f, 
                                  kx=self.splint_order,
                                  ky=self.splint_order)
        
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

        # Gaussian process prior.
        # Reshape parameters somewhat. 
        sedw = params.sedw.ravel()
        l = np.asarray([params.lp, params.llam])

        # Compute the covariance matrix. 
        sigma = self.diffmat / l
        sigma =  np.exp(-np.sum(sigma * sigma, axis=-1))
        sigma *= ETA_SQ # eta-sq is fixed now, but it may float in
                        # future versions.
        sigma += np.eye(sedw.size) * NUG # nug is fixed now, but it
                                         # may float in future
                                         # versions.
        mu = np.ones(sedw.size) # mean vector is 1 (no warping). 
        lp__ += stats.multivariate_normal.logpdf(sedw, mean=mu, cov=sigma)
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
        func = interp1d(x, y, kind=self.splint_order)
        return func
        
    def tpeak(self, params, retfunc=False):
        func = self.Lfunc(params)
        
        def minfunc(t):
            # objective function
            try:
                return -func(t) / 1e43
            except ValueError:
                return np.inf
    
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
            vec = ParamVec(params, self.nph, self.nl)
        except BoundsError as e:
            return -np.inf
        return self.logprior(vec) + self.loglike(vec)

    @property
    def D(self):
        return 4 + self.nph * self.nl
                         
def main(lc_filename, nph, outfile, nburn=NBURN, nsamp=NSAMP, nl=NL, 
         nwalkers=NWALKERS, nthreads=NTHREADS, exclude_bands=EXCLUDE_BANDS,
         dust_type=DUST_TYPE, rv_bintype=RV_BINTYPE, 
         splint_order=SPLINT_ORDER):
    
    # Fit a single light curve with the model.
    fc = FitContext(lc_filename=lc_filename, nph=nph, nl=nl,
                    exclude_bands=exclude_bands, dust_type=dust_type,
                    rv_bintype=rv_bintype, splint_order=splint_order)

    # create initial parameter vectors
    pvecs = list()

    diffs = fc.xstar[:, None] - fc.xstar[None, :]
    nmat = np.diag(np.ones(diffs.shape[0]) * NUG)
    
    for i in range(nwalkers):
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
    sampler = emcee.EnsembleSampler(nwalkers, fc.D, fc, threads=nthreads)

    # Get set up to collect the output of the sampler. 
    # Rename the output files if they already exist. 
     
    out = create_output_file(outfile, fc)

    with out:
        out.create_dataset('init_params', data=pvecs)

        burn = out.create_group('burn')
        samp = out.create_group('samples')

        initialize_hdf5_group(burn, fc, nburn, nwalkers)
        initialize_hdf5_group(samp, fc, nsamp, nwalkers)

        # Do burn-in.
        sgen_burn = sampler.sample(pvecs,
                                   iterations=nburn, 
                                   storechain=False)

        logging.info('beginning burn-in')
        for i, result in enumerate(sgen_burn):
            record(result, burn, fc, sampler, i)
            logging.info('burn-in iteration %d, med lnprob: %f',
                         i, np.median(result[1]))
            logging.info('median acceptance fraction = %f' % \
                         np.median(sampler.acceptance_fraction))
        logging.info('burn-in complete')
        
        # Reset the sampler
        sampler.reset()

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
            record(result, samp, fc, sampler, i)
            logging.info('sampling iteration %d, med lnprob: %f',
                         i, np.median(result[1]))
            logging.info('median acceptance fraction = %f' % \
                         np.median(sampler.acceptance_fraction))
        logging.info('sampling complete')
        

if __name__ == "__main__":
    
    # Create the argument parser. 
    parser = argparse.ArgumentParser()
    parser.add_argument('lc_filename', help='The name of the light ' \
                        'curve file to fit.', type=argparse.FileType('r'))
    parser.add_argument('nph', help='The number of phase points to use.',
                        type=int)
    parser.add_argument('outfile', help='The name of the hdf5 ' \
                        'file to store the MCMC results.', type=str)
    parser.add_argument('--logfile', help='The name of the MCMC logfile.',
                        default=LOGFILE, dest='logfile')
    parser.add_argument('--nburn', help='Number of burn-in iterations.',
                        default=NBURN, type=int, dest='nburn')
    parser.add_argument('--nsamp', help='Number of sampling iterations.',
                        default=NSAMP, type=int, dest='nsamp')
    parser.add_argument('--nl', help='Enables a regularly spaced wavelength ' \
                        'grid, and specifies the number of points to use.',
                        type=int, default=NL, dest='nl')
    parser.add_argument('--nwalkers', help='Number of walkers to use.',
                        type=int, default=NWALKERS, dest='nwalkers')
    parser.add_argument('--nthreads', help='Number of MCMC threads to use.',
                        type=int, default=NTHREADS, dest='nthreads')
    parser.add_argument('--exclude_bands', type=str, nargs='+', 
                        default=EXCLUDE_BANDS, help='Bandpasses to exclude ' \
                        'from the fit.', dest='exclude_bands')
    parser.add_argument('--dust_type', help='Reddening law to use for host ' \
                        'galaxy dust.', dest='dust_type', 
                        choices=['F99', 'OD94'], default=DUST_TYPE)
    parser.add_argument('--rv_bintype', help='Prior for host galaxy reddening' \
                        ' law.', dest='rv_bintype', 
                        default=RV_BINTYPE, choices=['gmm', 'uniform',
                                                     'binned'])
    parser.add_argument('--splint_order', help='Spline interpolation order.',
                        dest='splint_order', type=int,
                        default=SPLINT_ORDER, choices=[1,2,3])
    args = parser.parse_args()

    print args

    # Set the format of the log file. 
    lformat = '[%(asctime)s]: %(message)s'
    
    # TODO: Consider whether the following block should be moved to
    # module level?

    # If a logfile is specified...
    if args.logfile is not None:
        # ...log to it.
        logging.basicConfig(format=lformat,
                            filename=args.logfile, 
                            filemode='w',
                            level=logging.DEBUG)
    else:
        # ...else log to stdout. 
        logging.basicConfig(format=lformat,
                            level=logging.DEBUG)

    # Do business.
    main(lc_filename=args.lc_filename, nph=args.nph, outfile=args.outfile,
         nburn=args.nburn, nsamp=args.nsamp, nl=args.nl, 
         nwalkers=args.nwalkers, nthreads=args.nthreads,
         exclude_bands=args.exclude_bands, dust_type=args.dust_type,
         rv_bintype=args.rv_bintype, splint_order=args.splint_order)
