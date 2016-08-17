import sncosmo
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.interpolate import (RectBivariateSpline as Spline2d,
                               interp1d)
from copy import copy


__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Subclass of sncosmo.Model that implements ' \
              'warping with kernels.'


def bump_model(dust_type):
    model = sncosmo.Model(BumpSource(),
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'],
                          effects=[dust_type(), sncosmo.F99Dust()])
    return model


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
    
    def __init__(self, name, minwave, maxwave, minphase, maxphase, neg=False):
        self.name = name
        self._minwave = minwave
        self._maxwave = maxwave
        self._minphase = minphase
        self._maxphase = maxphase
        self._tophat = SmoothTophat(minwave, maxwave, 0.05)
        self.neg = neg

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
        lc /= lc.max() if not self.neg else lc.min() # normalize so
                                                     # max = 1 (same
                                                     # as amplitude of
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

    BUMPS = [Bump('UV', 1000., 3500., -7., 6),
             Bump('blue', 3500., 6900., -20, 20),
             Bump('i1', 6900., 9000., -12, 15),
             Bump('i2', 6900., 9000., 20, 32),
             Bump('y1', 9000., 11200., -10, 6),
             Bump('y2', 9000., 11200., 23., 38.),
             Bump('y3', 9000., 11200., 6, 23., neg=True),
             Bump('j1', 11200., 14000., -20, 15.),
             Bump('j2', 11200., 14000., 15.+6, 43.-6),
             Bump('j3', 11200., 14000., 5, 21., neg=True),
             Bump('h1', 14000., 19000.,-10., 6.),
             Bump('h2', 14000., 19000., 22.-2.5, 34.-2.5),
             Bump('h3', 14000., 19000., 5, 20., neg=True),
             Bump('k1', 19000., 25000., -7., 6.),
             Bump('k2', 19000., 25000., 20., 31.)]

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
            if bump.name == 'blue':
                self._parameters = np.concatenate((self._parameters, [0.]))
                self._param_names.append(bump.name + '_bump_slope')
                self.param_names_latex.append(bump.name + '_s')

    def minphase(self):
        return self._parameters[1] * self._phase[0]

    def maxphase(self):
        return self._parameters[1] * self._phase[-1]
    
    def _warp(self, phase, wave):
        warp = 1.
        for bump in self.bumps:
            if bump.name != 'blue':
                warp *= (1 + self.get(bump.name + '_bump_amp') * \
                             bump.kernel(phase / self.get('s'), wave))
            else:
                warp *= (1 + (self.get(bump.name + '_bump_amp') + \
                                  self.get(bump.name + '_bump_slope') * wave) * \
                             bump.kernel(phase / self.get('s'), wave))
        return warp

    def _flux(self, phase, wave):
        f = self._model_flux(phase / self._parameters[1], wave) * self._parameters[0]
        warp = self._warp(phase, wave)
        return f * warp


    
