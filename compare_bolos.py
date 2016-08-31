# compare the real bolometric light curve of 2011fe to the fit result

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import optimize, interpolate
from scipy import stats
from astropy import units as u
import pandas as pd
import emcee
from matplotlib import pyplot as plt

# 2011fe time series
ts = pd.read_csv('scripts/sn2011fe_1a.v3.2.dat', sep=' ', names=['phase', 'wave', 'flux'])

phase = np.unique(ts.phase)
wave = np.unique(ts.wave)
flux = ts.flux.reshape(phase.size, wave.size)

lc = []
# "measured" bolometric light curve 
for p, day in ts.groupby('phase'):
    lc.append(day.flux.sum() * 10)
lc = np.array(lc)

print lc

# priors
t0_prior = stats.uniform(13, 17)
Dl_prior = stats.norm(6.4, 0.5)

cm = lambda dl: (dl * u.Mpc).to(u.cm).value

# monte carlo estimate covariance matrix 
# on bolometric light 
# curve due to uncertainty in distance

Dl_samples = Dl_prior.rvs(1e4)
lcs = lc[None, :] * (4 * np.pi * (cm(Dl_samples)**2)[:, None])
sig = np.cov(lcs, rowvar=0)

fig,ax = plt.subplots(figsize=(10,5))
ax.errorbar(phase, lcs.mean(axis=0), yerr=lcs.std(axis=0), color='k', marker='.', 
            capsize=0, ls='None', label='2011fe (Amanullah+2015)')

ax.set_xlim(-20, 80)
ax.set_xlabel("Time since bolometric peak (days)")
ax.set_ylabel(r"$L$ (erg)")
ax.grid(True)

from bolomc import bolo
import pickle
import samples

lc, _, models = samples.models('scripts/2011fe.out')
stack = bolo.LCStack.from_models(models, dl=6.4)
ax = stack.plot(ax=ax)
fig.savefig('scripts/2011fe_bolo_comparison.pdf')

import sncosmo

fig = sncosmo.plot_lc(data=lc, model=models, fill_percentiles=(0.,50.,100.))
fig.savefig('2011fe.broadband.pdf')
