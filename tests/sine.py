#!/usr/bin/env python
# TODO:: Clean up this module. 

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Test problem with a uniform warping surface."

import sys
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
llam = 0.1 # Angstrom. TODO:: Verify this.
z = 0.02 
mwebv = 0.02


# Set up the test problem.

# Inject in a 2D sine curve. 

def sedw(p, l):
    """The injected warping function."""
    return 1 + 0.3 * np.sin((np.pi * p / lp) + (np.pi * np.log10(l) / llam)) 

test = problem.Problem(sedw, rv, ebv, z, mwebv)
data = test.data(100/8, exclude_bands=['csphd','cspjd','cspyd',
                                       'cspv3014','cspv3009'])

source = sncosmo.TimeSeriesSource(test.source._phase,
                                  test.source._wave,
                                  test.wsurf)

pickle.dump(source, open('answer.pkl','wb'))

fname = 'sine.pkl2'
pickle.dump(data, open(fname, 'wb'))

rv_prior = TruncNorm(0, np.inf, rv, 1.)
ebv_prior = TruncNorm(0, np.inf, ebv, 0.02)

fc = bolomc.TestProblemFitContext(lc_filename=fname, 
                                  nph=10, nl=10,
                                  mwebv=mwebv,
                                  ebv_prior=ebv_prior,
                                  rv_prior=rv_prior,
                                  splint_order=3)

nwalkers = fc.D * 2

pvecs = []
answer = np.asarray([sedw(*x) for x in fc.xstar])

for i in range(nwalkers):
    pvec = np.zeros(fc.D)
    pvec[:2] = [rv_prior.rvs(), ebv_prior.rvs()]
    pvec[2:] = np.random.uniform(size=fc.D - 2) * 4
    pvecs.append(pvec)

pvecs[0][0] = rv
pvecs[0][1] = ebv
pvecs[0][2:]= answer

# Some auxiliary data structures for animating the warping surface. 
surf2 = fc._create_model(bolomc.ParamVec(pvecs[0], fc.nph, fc.nl)).source._passed_flux / fc.hsiao._passed_flux
surf2[np.isnan(surf2)] = 0.
source2 = sncosmo.TimeSeriesSource(fc.hsiao._phase,
                                   fc.hsiao._wave,
                                   surf2)

pickle.dump(source2, open("init.pkl",'wb'))
bolomc.main(fc, 'sine.h52',
            1000, 1000,
            nwalkers=nwalkers,
            nthreads=4,
            fc_fname='sinefc.pkl2',
            logfile='sine.log2',
            pvecs=pvecs)
