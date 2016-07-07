
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Subclass of sncosmo.Model that implements ' \
              'warping with kernels.'

import sncosmo
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import RectBivariateSpline as Spline2d
from copy import copy
from scipy.linalg import pinv

class SmoothTophat(object):
    """A callable one-dimensional smooth tophat function with steepness
    parameter k.

    """
    def __init__(self, l, r, k):
        self.l = l # left inflection point
        self.r = r # right inflection point
        self.k = k # steepness parameter 
        
    def __call__(self, x):
        yr = 1. / (1. + np.exp(self.k*(x-self.r)))
        yl = 1. / (1. + np.exp(self.k*(self.l-x)))
        return yr * yl


class Gaussian(object):    
    """A one-dimensional Gaussian with amplitude 1, mean `mu` and scale
    `sigma`."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x - self.mu)**2 / (2 * self.sigma**2))


class Bump(object):
    
    def __init__(self, name, minwave, maxwave, minphase, maxphase):
        self.name = name
        self._minwave = minwave
        self._maxwave = maxwave
        self._minphase = minphase
        self._maxphase = maxphase
        self._tophat = SmoothTophat(minwave, maxwave, 0.05)

    def minphase(self):
        return self._minphase
        
    def maxphase(self):
        return self._maxphase
        
    def minwave(self):
        return self._minwave
        
    def maxwave(self):
        return self._maxwave

    def kernel(self, phase, wave):
        """Gaussian in time and smooth tophat in wavelength."""
        g = np.atleast_1d(self._gaussian(phase))
        t = np.atleast_1d(self._tophat(wave))
        result = g[:, None] * t[None, :]
        pslice = 0 if phase.ndim == 0 else slice(None)
        wslice = 0 if wave.ndim == 0 else slice(None)
        return result[pslice, wslice]

    def _fit_kernel(self, source, guess=None):

        """Fit the `mu` and `sigma` parameters of the gaussian (phase)
        component of the kernel.

        """
        wave_inds = np.logical_and(source._wave >= self.minwave(),
                                   source._wave <= self.maxwave())
        wave = source._wave[wave_inds]

        chisq_phases = np.linspace(self.minphase(), self.maxphase())
        
        if guess is None:
            guess=(.5*(self.minphase() + self.maxphase()), 5.)
        
        
        lc = source.flux(chisq_phases, wave).sum(-1)
        lc /= lc.max() # normalize so max = 1 (same as amplitude of
                       # gaussian)

        def chisq((mu, sigma)):
            pred = Gaussian(mu, sigma)(chisq_phases)
            
            # bounds checking 
            l = mu - sigma
            r = mu + sigma
            
            if r > self.maxphase() or l < self.minphase():
                return np.inf

            return np.sum((lc - pred)**2)
            
        opt, f, d = fmin_l_bfgs_b(chisq, guess, approx_grad=True)
        
        if d['warnflag'] == 0:
            self._gaussian = Gaussian(*opt)
        else:
            raise RuntimeError(d['warnflag'])

class HsiaoBump(Bump):
    
    def __init__(self, name):
        hsiao = sncosmo.get_source('hsiao')
        super(HsiaoBump, self).__init__(name, hsiao.minphase(),
                                        hsiao.maxphase(), 
                                        hsiao.minwave(),
                                        hsiao.maxwave())

    
class BumpSource(sncosmo.Source):

    """A single-component spectral time series model, that "stretches" in
    time.

    The spectral flux density of this model is given by

    .. math::

       F(t, \lambda) = A \\times M(t / s, \lambda)

    where _A_ is the amplitude and _s_ is the "stretch".

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_disp)`.
    """

    BUMPS = [Bump('UV', 1000., 3000., -20, 20),
             Bump('blue', 3000., 6900., -20, 20),
             Bump('i1', 6900., 9000., -20, 15),
             Bump('i2', 6900., 9000., 15, 42),
             Bump('y1', 9000., 11200., -20, 13.6),
             Bump('y2', 9000., 11200., 13.6, 43.),
             Bump('j1', 11200., 14000., -20, 15.),
             Bump('j2', 11200., 14000., 15., 43.),
             Bump('h1', 14000., 19000.,-20., 6.),
             Bump('h2', 14000., 19000., 15., 43.),
             Bump('k1', 19000., 25000., -20., 15.),
             Bump('k2', 19000., 25000., 15., 43.)]

    def __init__(self, name=None, version=None):

        self.name = name
        self.version = version
        hsiao = sncosmo.get_source("hsiao", version='3.0')        
        self._phase = hsiao._phase
        self._wave = hsiao._wave
        self._passed_flux = hsiao._passed_flux
        self._param_names = ['amplitude', 's']
        self.param_names_latex = ['A', 's']
        self._parameters = np.array([1., 1.])
        self._model_flux = Spline2d(self._phase, self._wave, 
                                    self._passed_flux, kx=2, ky=2)
        self.bumps = copy(self.BUMPS)
        
        for bump in self.bumps:
            bump._fit_kernel(hsiao)
            self._parameters = np.concatenate((self._parameters, [0.]))
            self._param_names.append(bump.name + '_bump_amp')
            self.param_names_latex.append(bump.name + '_A')

    def minphase(self):
        return self._parameters[1] * self._phase[0]

    def maxphase(self):
        return self._parameters[1] * self._phase[-1]
        
    def _flux(self, phase, wave):
        f = (self._parameters[0] *
             self._model_flux(phase / self._parameters[1], wave))
        for i, bump in enumerate(self.bumps):
            f *= (1 + self._parameters[i + 2] * \
                  bump.kernel(phase / self._parameters[1], wave))
        return f
    

class FreeBumpSource(sncosmo.Source):
    
    def __init__(self, nbumps, name=None, version=None):
        self.name = name
        self.version = version
        hsiao = sncosmo.get_source("hsiao", version='3.0')        
        self._phase = hsiao._phase
        self._wave = hsiao._wave
        self._passed_flux = hsiao._passed_flux
        self._param_names = ['amplitude', 's']
        self.param_names_latex = ['A', 's']
        self._parameters = np.array([1., 1.])
        self._model_flux = Spline2d(self._phase, self._wave, 
                                    self._passed_flux, kx=2, ky=2)
        self.nbumps = nbumps
        
        for i in range(self.nbumps):
            self._parameters = np.concatenate((self._parameters, [0.] * 6))
            self._param_names += map(lambda pref: pref + str(i), ['mu_phase_',
                                                                  'mu_wave_',
                                                                  'sig_phase_',
                                                                  'sig_wave_',
                                                                  'cov_',
                                                                  'amp_'])

            self.param_names_latex += map(lambda tok: tok % i, ['\mu_{p,%d}',
                                                                '\mu_{\lambda, %d}',
                                                                '\sigma_{p,%d}',
                                                                '\sigma_{\lambda, %d}',
                                                                '\sigma_{p \lambda, %d}',
                                                                'A_%d'])

    def minphase(self):
        return self._parameters[1] * self._phase[0]

    def maxphase(self):
        return self._parameters[1] * self._phase[-1]
        
    def _unscale(self, scaled_params):
        """scaled = A unscaled + B"""
        upper = np.array([self._phase[-1], np.log10(self._wave[-1]),
                          0.5 * (self._phase[-1] - self._phase[0]),
                          0.5 * (np.log10(self._wave[-1]) - np.log10(self._wave[0])),
                          1., 2.])
        lower = np.array([self._phase[0], np.log10(self._wave[0]), 
                          0., 0., 0, -2.])
        
        diff = upper - lower 
        A = 2 / diff
        B = 1 - (2 * upper / diff)
        unscaled = (scaled_params - B) / A
        unscaled[4] = unscaled[2] * unscaled[3] * scaled_params[4]
        return unscaled
        
        
    def _flux(self, phase, wave):
        f = (self._parameters[0] *
             self._model_flux(phase / self._parameters[1], wave))
        
        p = np.asarray(phase)
        w = np.log10(np.asarray(wave))
        
        ndim_p = p.ndim
        ndim_w = w.ndim
        
        p = np.atleast_1d(p)
        w = np.atleast_1d(w)
        
        grid = np.asarray([[(pp, ww) for ww in w] for pp in p])
        flatgrid = grid.reshape(-1, 2)
        
        pslice = 0 if ndim_p == 0 else slice(None)
        wslice = 0 if ndim_w == 0 else slice(None)

        for i in range(self.nbumps):
            if i == self.nbumps - 1:
                scaled_params = self.parameters[(-self.nbumps + i) * 6:]
            else:
                scaled_params = self.parameters[(-self.nbumps + i) * 6:(-self.nbumps + i + 1) * 6]
            params = self._unscale(scaled_params)
            cov = np.array([[params[2]**2, params[4]],[params[4], params[3]**2]])
            invcov = pinv(cov)
            mu = params[:2]
            devs = flatgrid - mu
            kern = np.asarray([np.exp(-row.dot(invcov).dot(row)) for row in devs]) * params[-1] 
            kern = kern.reshape(p.shape + w.shape)
            kern = kern[pslice, wslice]
            f *= (1 + kern)

        return f
