
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Light curve plotting utilities.'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sncosmo

sns.set_style('ticks')

class LCStack(object):
    
    @classmethod
    def from_file(cls, f):
        L = np.genfromtxt(f)
        phase = sncosmo.get_source('hsiao', version='3.0')._phase
        L = L[:, 2:].reshape(tuple(L[:, :2].astype('<i8').max(0) + 1) + 
                             (phase.size,))
        return cls(phase, L)

    def __init__(self, phase, L, name=None):
        self.phase = phase
        self.L = np.atleast_2d(L)
        self.name = name

    def plot(self):
        fig, ax = plt.subplots()
        for row in self.L.reshape(-1, 106):
            ax.plot(self.phase, row, 'k', alpha=0.05)
        ax.set_xlabel('phase (days)')
        ax.set_ylabel('flux (cgs)')
        ax.set_title(self.name)
        sns.despine(ax=ax)
        return ax
