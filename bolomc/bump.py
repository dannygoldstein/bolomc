
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Subclass of sncosmo.Model that implements ' \
              'warping with kernels.'

import sncosmo
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


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
        self._name = name
        self._minwave = minwave
        self._maxwave = maxwave
        self._minphase = minphase
        self._maxphase = maxphase
        self._tophat = smooth_tophat(minwave, maxwave, 2.)

    def name(self):
        return self._name
        
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

    def affects(self, phase, wave, nsigma=4):
        """True if `phase` and `wave` are within the area of influence of the
        kernel.

        """
        th_l = wave >= self._tophat.l
        th_r = wave <= self._tophat.r
        
        gs_l = phase >= self._gaussian.mu - nsigma * self._gaussian.sigma
        gs_r = phase <= self._gaussian.mu + nsigma * self._gaussian.sigma
        
        return th_l and th_r and gs_l and gs_r


    def _fit_kernel(self, source, 
                    guess=(.5*(self.minphase() + self.maxphase()), 5.)):

        """Fit the `mu` and `sigma` parameters of the gaussian (phase)
        component of the kernel.

        """
        wave_inds = np.logical_and(source._wave >= self.minwave(),
                                   source._wave <= self.maxwave())
        wave = source._wave[wave_inds]

        chisq_phases = np.linspace(self.minphase(), self.maxphase())
        
        lc = source.flux(chisq_phases, wave).sum(-1)
        lc /= lc.max() # normalize so max = 1 (same as amplitude of
                       # gaussian)

        def chisq((mu, sigma)):
            pred = gaussian(mu, sigma)(chisq_phases)
            
            # bounds checking 
            l = mu - 2 * sigma
            r = mu + 2 * sigma
            
            if r > self.maxphase() or l < self.minphase():
                return np.inf

            return np.sum((lc - pred)**2)
            
        x, f, d = fmin_l_bfgs_b(chisq, guess, approx_grad=True)
        if d['warnflag'] == 0:
            self._gaussian = gaussian(*x)
        else:
            raise RuntimeError(d['warnflag'])

    
class BumpSource(sncosmo.StretchSource):

    def __init__(self, phase, wave, flux, bumps, name=None, version=None):
        super(BumpSource, self).__init__(phase, wave, flux, 
                                         name=name, version=version)
        
        self.bumps = bumps
        for bump in self.bumps:
            bump._fit_kernel(self)
            self._parameters.append(1.)
            self._param_names.append(bump.name + '_bump_amp')
            self.param_names_latex.append(bump.name + '_A')
        
    def flux(self, phase, wave):
        f = super(BumpSource, self).flux(phase, wave)
        for bump in self.bumps:
            f *= (1 + bump.kernel(phase, wave))
        return f
