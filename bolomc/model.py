#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

__all__ = ['TestProblemFitContext', 'CSPFitContext', 
           'main', 'main_csp', 'restart']

import os
import sys
import abc
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
import george

import logging
import argparse
import pickle

######################################################
# CONSTANTS ##########################################
######################################################

h = 6.62606885e-27 # erg s
c = 2.99792458e10  # cm / s
AA_TO_CM = 1e-8 # dimensionless
ETA_SQ = 0.1
NUG = 1e-7 
magsys = sncosmo.get_magsystem('csp')
nmersenne = 624 # size of MT state vector
# Set the format of the log file. 
lformat = '[%(asctime)s]: %(message)s'

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
FC_FNAME = None
PVECS = None

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

    # Save the random state. 
    group['rstate1'][i] = rstate[1]
    group['rstate2'][i] = rstate[2]
    group['rstate3'][i] = rstate[3]
    group['rstate4'][i] = rstate[4]
    
    # Save the acceptance fraction. 
    group['afrac'][i] = sampler.acceptance_fraction
    
    # Save the current iteration number.
    group['last_index_filled'][()] = i

def rename_output_file(name):
    newname = name + '.old'
    if os.path.exists(newname):
        rename_output_file(newname)
    os.rename(name, newname)

def create_output_file(fname, clobber=False):
    if os.path.exists(fname) and not clobber:
        rename_output_file(fname)
    f = h5py.File(fname, 'w')        
    return f

def initialize_hdf5_group(group, fc, nsamp, nwalkers):

    bs = (nsamp, nwalkers, fc.hsiao._phase.shape[0])
    afs = (nsamp, nwalkers)
    pars = (nsamp, nwalkers, fc.D)
    probs = (nsamp, nwalkers)
    states = (nsamp, nmersenne)

    # Initialize the result arrays. 
    group.create_dataset('params', pars, dtype='float64')
    group.create_dataset('prob', probs, dtype='float64')
    group.create_dataset('afrac', afs, dtype='float64')
    group.create_dataset('bolo', bs, dtype='float64')
    
    # Initialize the random state arrays. 
    group.create_dataset('rstate1', states, dtype='uint32')
    group.create_dataset('rstate2', (nsamp,), dtype=int)
    group.create_dataset('rstate3', (nsamp,), dtype=int)
    group.create_dataset('rstate4', (nsamp,), dtype='float64')

    group['niter'] = nsamp
    group['last_index_filled'] = -1

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
    
    def __init__(self, lc_filename, nph, lp, llam, nl=NL, dust_type=DUST_TYPE,
                 exclude_bands=EXCLUDE_BANDS, rv_bintype=RV_BINTYPE,
                 splint_order=SPLINT_ORDER):

        # Create the FitContext attributes.
        self.dust_type = dust(dust_type)
        self.exclude_bands = exclude_bands
        self.splint_order = splint_order
        self.nph = nph
        self.nl = nl
        self.passed_nl = nl
        self.lc_filename = lc_filename
        self.lp = lp
        self.llam = llam

        # self.lc is now an accessible attribute

        self.lc['wave_eff'] = map(filter_to_wave_eff, self.lc['filter'])
        self.lc.sort(['mjd', 'wave_eff'])

        self.rv_bintype = rv_bintype

        self.bands = [sncosmo.get_bandpass(band) for
                      band in np.unique(self.lc['filter']) if
                      band not in self.exclude_bands]

        # Load Hsiao SED into memory here so it doesn't have to be
        # loaded every time _create_model is called.

        self.hsiao = sncosmo.get_source('hsiao', version='3.0')
        
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
        guess_mod.set(hostebv=self.ebv_prior.mean)
        guess_mod.set(hostr_v=self.rv_prior.mean)
        guess_mod.set(mwebv=self.mwebv)
        
        # To avoid egregiously bad guesses of t0, we will give sncosmo
        # an initial guess for the mjd of B-band maximum by scraping
        # the results of SNooPy fits to the photometry from the CSP
        # website (read here from a file).
        
        try:
            t0 = get_t0(self.lc.meta['name'])
        except KeyError:
            # If the SN does not have a t0 estimate from CSP, fit it.
            res, fitted_model = sncosmo.fit_lc(self.lc,
                                               guess_mod,
                                               ['amplitude', 
                                                't0'])
            t0 = res['parameters'][1]

        else:
            guess_mod.set(t0=t0)
            res, fitted_model = sncosmo.fit_lc(self.lc,
                                               guess_mod,
                                               ['amplitude'])            

        if not res['success']:
            raise FitError(res['message'])

        self.amplitude = res['parameters'][2]
        self.t0 = t0

        # Fit the model. 

        self.hsiao_binw = np.gradient(self.hsiao._wave)


        # This is defined here and used repeatedly in the logprior
        # calculation.

        self.x_hsiao = np.asarray(list(product(self.hsiao._phase, 
                                               self.hsiao._wave)))

        self.kernel = ETA_SQ * george.kernels.ExpSquaredKernel([self.lp, 
                                                                self.llam], 
                                                               ndim=2)
        self.gp = george.GP(self.kernel, mean=1)
        self.gp.compute(self.xstar, yerr=NUG)

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, x):
        try:
            self.lc['mjd'] = self.lc['mjd'] + self.t0
        except AttributeError:
            pass
        self._t0 = x
        self.lc['mjd'] = self.lc['mjd'] - self.t0
    
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
        lp__ += self.gp.lnlikelihood(sedw)

        return lp__

    def loglike(self, params):
        """Compute loglike(params)."""
        
        lp__ = 0
        model = self._create_model(params)
        flux = model.bandflux(self.lc['filter'],
                              self.lc['mjd'],
                              zp=self.lc['zp'],
                              zpsys=self.lc['zpsys'])

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
            distfac = 4 * np.pi * Planck13.luminosity_distance(
                self.lc.meta['zcmb']).to(u.cm).value**2
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
        return 2 + self.nph * self.nl

class CSPFitContext(FitContext):

    @property
    def mwebv(self):
        try:
            return self._mwebv
        except AttributeError:
            self._mwebv, _ = get_mwebv(self.lc.meta['name'])
            return self._mwebv

    @property
    def ebv_prior(self):
        try:
            return self._ebv_prior
        except AttributeError:
            self.host_ebv, self.host_ebv_err = get_hostebv(self.lc.meta['name'])
            self._ebv_prior = TruncNorm(0., np.inf, 
                                        self.host_ebv,
                                        self.host_ebv_err)
            return self._ebv_prior
            
    @property
    def rv_prior(self):
        try:
            return self._rv_prior
        except AttributeError:
            self._rv_prior = get_hostrv_prior(self.lc.meta['name'],
                                              self.rv_bintype,
                                              self.dust_type)
            return self._rv_prior

    @property
    def lc(self):
        try:
            return self._lc
        except AttributeError:
            self._lc = sncosmo.read_lc(lc_filename, format='csp')
            return self._lc
            
class TestProblemFitContext(FitContext):
    
    @property
    def mwebv(self):
        return self._mwebv
    
    @property
    def ebv_prior(self):
        return self._ebv_prior
    
    @property
    def rv_prior(self):
        return self._rv_prior
                
    @property
    def lc(self):
        try:
            return self._lc
        except AttributeError:
            self._lc = pickle.load(open(self.lc_filename,'rb'))
            return self._lc
        

    def __init__(self, lc_filename, nph, lp, llam, mwebv, ebv_prior, 
                 rv_prior, nl=NL, dust_type=DUST_TYPE,
                 exclude_bands=EXCLUDE_BANDS, rv_bintype=RV_BINTYPE,
                 splint_order=SPLINT_ORDER):
        
        self._mwebv = mwebv
        self._ebv_prior = ebv_prior
        self._rv_prior = rv_prior

        super(TestProblemFitContext, self).__init__(lc_filename, nph, lp, llam, 
                                                    nl=nl,
                                                    dust_type=dust_type,
                                                    exclude_bands=exclude_bands,
                                                    rv_bintype=rv_bintype,
                                                    splint_order=splint_order)
            

def generate_pvec(fc):
    """Generate initial parameter vectors for the FitContext `fc`."""


    # TODO: Implement more principled initialization for warping
    # function parameters.
    
    rv = fc.rv_prior.rvs()
    ebv = fc.ebv_prior.rvs()
        
    # Ensure all initial warping function values are positive.
    sedw = np.ones(fc.xstar.shape[0]) * -1
    while (sedw < 0).any():
        sedw = fc.gp.sample()
        
    return np.concatenate(([rv, ebv], sedw))
    

def main(fc, outfile, nburn=NBURN, nsamp=NSAMP,
         nwalkers=NWALKERS, nthreads=NTHREADS, dust_type=DUST_TYPE,
         fc_fname=FC_FNAME, logfile=LOGFILE, pvecs=PVECS):

    # If a logfile is specified...
    if logfile is not None:
        # ...log to it.
        logging.basicConfig(format=lformat,
                            filename=logfile, 
                            filemode='w',
                            level=logging.DEBUG)
    else:
        # ...else log to stdout. 
        logging.basicConfig(format=lformat,
                            level=logging.DEBUG)

    # Create initial parameter vectors. 
    
    if pvecs is None:
        pvecs = [generate_pvec(fc) for i in range(nwalkers)]
    
    # Set up the sampler. 
    sampler = emcee.EnsembleSampler(nwalkers, fc.D, fc, threads=nthreads)

    # Get set up to collect the output of the sampler. 
    # Rename the output files if they already exist. 
     
    out = create_output_file(outfile)


    with out:
        out.create_dataset('init_params', data=pvecs)
        out.create_dataset('current_stage', (), dtype=int)
        
        
        # Pickle the FitContext first. 
        if fc_fname is not None:
            pickle.dump(fc, open(fc_fname, 'wb'))
            out['fc_fname'] = fc_fname
        
        # These read emcee configuration settings. 

        out['nwalkers'] = nwalkers
        out['nthreads'] = nthreads

                
        burn = out.create_group('burn')
        samp = out.create_group('samples')

        initialize_hdf5_group(burn, fc, nburn, nwalkers)
        initialize_hdf5_group(samp, fc, nsamp, nwalkers)
        
        # Do burn-in.
        
        out["current_stage"][()] = 0 # 0 = burn in, 1 = sampling

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
        
        out["current_stage"][()] = 1 # 0 = burn-in, 1 = sampling

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

             
def main_csp(lc_filename, nph, outfile, nburn=NBURN, nsamp=NSAMP, nl=NL, 
             nwalkers=NWALKERS, nthreads=NTHREADS, exclude_bands=EXCLUDE_BANDS,
             dust_type=DUST_TYPE, rv_bintype=RV_BINTYPE, 
             splint_order=SPLINT_ORDER, fc_fname=FC_FNAME, 
             logfile=LOGFILE):
    
    # Fit a single light curve with the model.
    fc = CSPFitContext(lc_filename=lc_filename, nph=nph, nl=nl,
                       exclude_bands=exclude_bands, dust_type=dust_type,
                       rv_bintype=rv_bintype, splint_order=splint_order)
    
    main(fc, outfile, nburn=nburn, nsamp=nsamp, nwalkers=nwalkers,
         nthreads=nthreads, dust_type=dust_type, fc_name=fc_name,
         logfile=logfile)
    

def restart(checkpoint_filename, iteration=None, stage=None, 
            logfile=LOGFILE):
    """Restart an MCMC run from an unfinished HDF5 file."""

    
    # If a logfile is specified...
    if logfile is not None:
        # ...log to it.
        logging.basicConfig(format=lformat,
                            filename=logfile, 
                            filemode='w',
                            level=logging.DEBUG)
    else:
        # ...else log to stdout. 
        logging.basicConfig(format=lformat,
                            level=logging.DEBUG)
    
    f = h5py.File(checkpoint_filename, 'a')
    fc_fname = f['fc_fname'][()]

    fc = pickle.load(open(fc_fname, 'rb')) 

    with f:
        
        # Are we burning or sampling? 

        if stage == 'burn':
            cs = f['burn']
        elif stage == 'samples':
            cs = f['samples']
        elif stage is None:
            stage_code = f["current_stage"][()]
            if stage_code == 0:
                cs = f['burn']
            elif stage_code == 1:
                cs = f['samples']
            else:
                raise ValueError('Invalid stage %s, in HDF5 file, must be one of '
                                 '[burn, samples, None]' % stage_string)
        else:
            raise ValueError('Invalid stage %s, must be one of '
                             '[burn, samples, None]' % stage)

        # Take the latest iteration for this group if a restart is not
        # specified, else use the given value.

        if iteration is None:
            i = cs["last_index_filled"][()]
        else:
            i = iteration
        
        # Read MCMC configuration parameters.
        nwalkers = f['nwalkers'][()]
        nthreads = f['nthreads'][()]

        # Read the random state.
        rstate = ('MT19937', # See
                             # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/
                             # numpy.random.get_state.html
                  cs['rstate1'][i],
                  cs['rstate2'][i],
                  cs['rstate3'][i],
                  cs['rstate4'][i])
        
        # Read the initial positions of the walkers. 
        pos = cs['params'][i]
    
        # And probabilities. 
        prob = cs['prob'][i]
        
        # And the sampler. 
        sampler = emcee.EnsembleSampler(nwalkers, fc.D, fc, threads=nthreads)        
        # Restarting during the burn-in stage is a bit different from
        # restarting during the sampling stage, because we have to
        # transition to the sampling stage afterward. 
        
        startinburn = 'burn' in cs.name
        burn = f['burn']
        samp = f['samples']

        if startinburn: 

            nburn = f['burn']['niter'][()] - i - 1
            nsamp = f['samples']['niter'][()]

            # 0 = burn-in, 1 = sampling
            f["current_stage"][()] = 0

            sgen_burn = sampler.sample(pos, 
                                       lnprob0=prob,
                                       rstate0=rstate,
                                       iterations=nburn,
                                       storechain=False)
            
            logging.info('resuming burn-in')
            for j, result in enumerate(sgen_burn):
                record(result, burn, fc, sampler, i + j + 1)
                logging.info('burn-in iteration %d, med lnprob: %f',
                             i + j + 1, np.median(result[1]))
                logging.info('median acceptance fraction = %f' % \
                             np.median(sampler.acceptance_fraction))
            logging.info('burn-in complete')

            # Set up sample generator.
            pos, prob, rstate = result

            # Reset the sampler
            sampler.reset()
            
        else:
            nsamp = f['samples']['niter'][()] - i 
            
        # 0 = burn-in, 1 = sampling
        f["current_stage"][()] = 1

        sgen = sampler.sample(pos,
                              iterations=nsamp,
                              rstate0=rstate,
                              lnprob0=prob,
                              storechain=False)

        logging.info('performing sampling')
        # Sample and record the output. 
        for j, result in enumerate(sgen):
            record(result, samp, fc, sampler, i + j + 1)
            logging.info('sampling iteration %d, med lnprob: %f',
                         i + j + 1, np.median(result[1]))
            logging.info('median acceptance fraction = %f' % \
                         np.median(sampler.acceptance_fraction))
        logging.info('sampling complete')
            

if __name__ == "__main__":
    
    # Create the argument parser and subparsers.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')

    # This parser will handle starting new MCMC runs. 
    primary_parser = subparsers.add_parser('run', help='Start a new MCMC.')
    
    # This simpler parser will handle restarting unfinished MCMC runs. 
    checkpoint_parser = subparsers.add_parser('restart', 
                                              help='Restart an unfinished MCMC run ' \
                                              'from an HDF5 file.')
    
    # Arguments for the `run` parser, which handles starting new MCMC runs. 
    primary_parser.add_argument('lc_filename', help='The name of the light ' \
                        'curve file to fit.', type=argparse.FileType('r'))
    primary_parser.add_argument('nph', help='The number of phase points to use.',
                        type=int)
    primary_parser.add_argument('outfile', help='The name of the hdf5 ' \
                        'file to store the MCMC results.', type=str)
    primary_parser.add_argument('--logfile', help='The name of the MCMC logfile.',
                        default=LOGFILE, dest='logfile')
    primary_parser.add_argument('--nburn', help='Number of burn-in iterations.',
                        default=NBURN, type=int, dest='nburn')
    primary_parser.add_argument('--nsamp', help='Number of sampling iterations.',
                        default=NSAMP, type=int, dest='nsamp')
    primary_parser.add_argument('--nl', help='Enables a regularly spaced wavelength ' \
                        'grid, and specifies the number of points to use.',
                        type=int, default=NL, dest='nl')
    primary_parser.add_argument('--nwalkers', help='Number of walkers to use.',
                        type=int, default=NWALKERS, dest='nwalkers')
    primary_parser.add_argument('--nthreads', help='Number of MCMC threads to use.',
                        type=int, default=NTHREADS, dest='nthreads')
    primary_parser.add_argument('--exclude_bands', type=str, nargs='+', 
                        default=EXCLUDE_BANDS, help='Bandpasses to exclude ' \
                        'from the fit.', dest='exclude_bands')
    primary_parser.add_argument('--dust_type', help='Reddening law to use for host ' \
                        'galaxy dust.', dest='dust_type', 
                        choices=['F99', 'OD94'], default=DUST_TYPE)
    primary_parser.add_argument('--rv_bintype', help='Prior for host galaxy reddening' \
                        ' law.', dest='rv_bintype', 
                        default=RV_BINTYPE, choices=['gmm', 'uniform',
                                                     'binned'])
    primary_parser.add_argument('--splint_order', help='Spline interpolation order.',
                        dest='splint_order', type=int,
                        default=SPLINT_ORDER, choices=[1,2,3])
    primary_parser.add_argument('--fc_fname', help='Pickle the fitcontext to this file.',
                                type='str', default=FC_FNAME)
    
    # Arguments for the checkpoint restart parser.     
    checkpoint_parser.add_argument('checkpoint_filename', help='The HDF5 file containing the ' \
                                   'results and settings of the run to restart.', type=str)
    checkpoint_parser.add_argument('--iteration', help='The iteration (0-based) to restart fr' \
                                   'om. If not specified, uses the next unfilled iteration fr' \
                                   'om the specified stage.', default=None, type=int)
    checkpoint_parser.add_argument('--stage', help='The stage to restart from. If not specifi' \
                                   'ed, uses the stage of the latest unfilled iteration.', 
                                   default=None, type=str, choices=['burn','samples'])
    checkpoint_parser.add_argument('--logfile', help='The name of the MCMC logfile.',
                                   default=LOGFILE, dest='logfile')

    args = parser.parse_args()

    # Do business.
    if args.subparser_name == 'restart':
        restart(args.checkpoint_filename, iteration=args.iteration,
                stage=args.stage, logfile=args.logfile)
    elif args.subparser_name == 'run':
        main_csp(lc_filename=args.lc_filename, nph=args.nph, outfile=args.outfile,
             nburn=args.nburn, nsamp=args.nsamp, nl=args.nl, 
             nwalkers=args.nwalkers, nthreads=args.nthreads,
             exclude_bands=args.exclude_bands, dust_type=args.dust_type,
             rv_bintype=args.rv_bintype, splint_order=args.splint_order,
             fc_fname=args.fc_fname, logfile=args.logfile)
