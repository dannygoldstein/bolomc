#!/usr/bin/env python

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Test problem with a uniform warping surface."

import matplotlib
matplotlib.use("Agg")
import bolomc
import sncosmo
import problem
import numpy as np
import pickle

from bolomc.distributions import TruncNorm

# True parameters.
rv = 3.1 # Host.
ebv = 0.1 # Host.
lp = 20 # Days. TODO:: Verify this. 
llam = 1500 # Angstrom. TODO:: Verify this.
z = 0.02 
mwebv = 0.02

def sedw(p, l):
    return 1 + 0.1 * np.sin((np.pi * p / lp) + (np.pi * l / llam)) 

test = problem.Problem(sedw, rv, ebv, lp, llam, z, mwebv)
data = test.data(100/8, exclude_bands=['csphd','cspjd','cspyd',
                                       'cspv3014','cspv3009'])

fname = 'sine.pkl'
pickle.dump(data, open(fname, 'wb'))

llam_prior = TruncNorm(0, np.inf, 1500/2., 200)
lp_prior = TruncNorm(0, np.inf, 10, 5.)
rv_prior = TruncNorm(0, np.inf, 3.1, 1.)
ebv_prior = TruncNorm(0, np.inf, 0.1, 0.02)

fc = bolomc.TestProblemFitContext(lc_filename=fname, 
                                  nph=10, mwebv=mwebv,
                                  ebv_prior=ebv_prior,
                                  rv_prior=rv_prior,
                                  llam_prior=llam_prior,
                                  lp_prior=lp_prior, 
                                  splint_order=1)

nwalkers = 300

pvecs = list()
for i in range(nwalkers):
    pvec = bolomc.model.generate_pvec(fc)
    pvec[4:] = np.asarray([sedw(*x) for x in fc.xstar])
    pvec[4:] += np.random.normal(0, 0.02, size=fc.np * fc.nl)
    pvecs.append(pvec)

bolomc.main(fc, 'sine.h5',
            1000, 1000,
            nwalkers=nwalkers,
            nthreads=2,
            fc_fname='sinefc.pkl',
            logfile='sine.log',
            pvecs=pvecs)
