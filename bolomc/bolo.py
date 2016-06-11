
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Light curve plotting utilities.'

import h5py
import numpy as np
import sncosmo

class LCStack(object):
    
    @classmethod
    def from_hdf5(cls, f):
        f = h5py.File(f)
        if f['current_stage'][()]:
            try:
                L = np.vstack(f['samples']['bolo'][[0, 49]])
            except:
                i = f['samples']['last_index_filled'][()]
                L = f['samples']['bolo'][i]
        else:
            i = f['burn']['last_index_filled'][()]
            L = f['burn']['bolo'][i]

        # for now
        phase = sncosmo.get_source('hsiao')._phase
        return cls(phase, L)

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
