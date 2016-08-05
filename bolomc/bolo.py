import h5py
import numpy as np
import sncosmo
from astropy.cosmology import Planck13
from scipy.optimize import minimize
from scipy.interpolate import interp1d

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Light curve plotting utilities.'


def bolometric(model, luminosity=True):
    phase = model.source._phase
    flux = np.sum(model.source.flux(phase, model.source._wave) * dwave, axis=1)
    if luminosity:
        z = model.get('z')
        dl = Planck13.luminosity_distance(z).to('cm').value
        L = 4 * np.pi * flux * dl * dl
        return L
    return flux

def Lfunc(model):
    y = bolometric(model)
    func = interp1d(x, y, kind='cubic')
    return func

def tpeak(model, retfunc=False):
    func = Lfunc(model)

    def minfunc(t):
        # objective function
        try:
            return -func(t) / 1e43
        except ValueError:
            return np.inf

    res = minimize(minfunc, 0.)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x if not retfunc else (res.x, func)

def Lpeak(model):
    tpeak, func = tpeak(model, retfunc=True)
    return func(tpeak)

def dm15(model):
    tpeak, func = tpeak(model, retfunc=True)
    lpeak = func(tpeak)
    l15 = func(tpeak + 15)
    return 2.5 * np.log10(lpeak / l15)


class LCStack(object):
    
    @classmethod
    def from_models(cls, models, name=None):
        # check to see if all models have the same phase grid
        phase0 = models[0].source._phase
        if not all([m._phase == phase[0] for m in models]):
            raise ValueError("all models need to have the same phase grid.")

        # make the light curves
        bolos = map(bolometric, models)

        # return the LCStack
        return cls(phase0, bolos, name=name)

    def __init__(self, phase, L, name=None):
        self.phase = phase
        self.L = np.atleast_2d(L)
        self.name = name

    def plot(self, ax=None):

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('ticks')
        
        if ax is None:
            fig, ax = plt.subplots()
        
        median_L = np.median(self.L, axis=0)
        L_upper = np.percentile(self.L, 50 + 68 / 2., axis=0)
        L_lower = np.percentile(self.L, 50 - 68 / 2., axis=0)
        
        ax.plot(self.phase, median_L, 'k', lw=1.3)
        ax.fill_between(self.phase, L_lower, L_upper, color='k', alpha=0.4)
        
        ax.set_xlabel('phase (days)')
        ax.set_ylabel('L (erg / s)')
        sns.despine(ax=ax)
        return ax
