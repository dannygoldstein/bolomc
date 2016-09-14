import h5py
import numpy as np
import sncosmo
from astropy.cosmology import Planck13
from astropy import units as u
from scipy.optimize import minimize
from scipy.interpolate import interp1d

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Light curve plotting utilities.'


def bolometric(model, luminosity=True, dl=None):
    phase = model.source._phase
    dwave = np.gradient(model.source._wave)
    flux = np.sum(model.source.flux(phase, model.source._wave) * dwave, axis=1)
    if luminosity:
        z = model.get('z')
        if dl is None:
            dl = Planck13.luminosity_distance(z).to('cm').value
        else:
            dl = (dl * u.Mpc).to('cm').value
        L = 4 * np.pi * flux * dl * dl
        return L
    return flux

def Lfunc(model):
    x = model.source._phase
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
    tp, func = tpeak(model, retfunc=True)
    return func(tp)

def dm15(model):
    tp, func = tpeak(model, retfunc=True)
    lpeak = func(tp)
    l15 = func(tp + 15)
    return 2.5 * np.log10(lpeak / l15)


class LCStack(object):
    
    @classmethod
    def from_models(cls, models, name=None, dl=None):
        # check to see if all models have the same phase grid
        phase0 = models[0].source._phase
        if not np.all([m.source._phase == phase0 for m in models]):
            raise ValueError("all models need to have the same phase grid.")

        # make the light curves
        bolos = [bolometric(model, dl=dl) for model in models]
        return cls(phase0, bolos, name=name)

    def __init__(self, phase, L, name=None):
        self.phase = phase
        self.L = np.atleast_2d(L)
        self.name = name

    def plot(self, ax=None, error=True, color=False, peak=False):

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        sns.set_style('ticks')
        
        if ax is None:
            fig, ax = plt.subplots()
        
        median_L = np.median(self.L, axis=0)
        L_upper = np.percentile(self.L, 50 + 68 / 2., axis=0)
        L_lower = np.percentile(self.L, 50 - 68 / 2., axis=0)

        func = interp1d(self.phase, median_L, kind='cubic')
        def minfunc(t):
            # objective function
            try:
                return -func(t) / 1e43
            except ValueError:
                return np.inf

        res = minimize(minfunc, 0.)
        if not res.success:
            raise RuntimeError(res.message)
        tp = res.x 
        Lmax = func(tp)
        L15 = func(tp + 15.)
        dm15 = -2.5 * np.log10(L15 / Lmax)


        if color:
            # color code by dm15
            jet = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=0.4, vmax=1.3)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            c = scalarMap.to_rgba(dm15)[0]
        else:
            c = 'k'

        ax.plot(self.phase, median_L/1e43, color=c, lw=1.3)
        
        if peak:
            ax.plot([tp], [Lmax/1e43], linestyle="None", marker='.', color='r')
        
        if error: 
            ax.fill_between(self.phase, L_lower/1e43, L_upper/1e43, color='k', alpha=0.4)
        
        ax.set_xlabel('phase (days)')
        ax.set_ylabel('L ($10^{43}$ erg / s)')
        sns.despine(ax=ax)
        return ax if not color else (ax, scalarMap, dm15)
