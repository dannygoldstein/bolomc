#!/usr/bin/env python

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

def sedw(p, l):
    return 1 + 0.3 * np.sin((np.pi * p / lp) + (np.pi * np.log10(l) / llam)) 

test = problem.Problem(sedw, rv, ebv, lp, llam, z, mwebv)
data = test.data(100/8, exclude_bands=['csphd','cspjd','cspyd',
                                       'cspv3014','cspv3009'])

source = sncosmo.TimeSeriesSource(test.source._phase,
                                  test.source._wave,
                                  test.wsurf)

pickle.dump(source, open('answer.pkl','wb'))

fname = 'sine.pkl2'
pickle.dump(data, open(fname, 'wb'))

llam_prior = TruncNorm(-np.inf, np.inf, llam, 0.03)
lp_prior = TruncNorm(0, np.inf, lp, 5.)
rv_prior = TruncNorm(0, np.inf, rv, 1.)
ebv_prior = TruncNorm(0, np.inf, ebv, 0.02)

fc = bolomc.TestProblemFitContext(lc_filename=fname, 
                                  nph=25, nl=40,
                                  mwebv=mwebv,
                                  llam_prior=llam_prior,
                                  lp_prior=lp_prior,
                                  ebv_prior=ebv_prior,
                                  rv_prior=rv_prior,
                                  splint_order=3)

nwalkers = fc.D * 2

pvecs = []
for i in range(nwalkers):
    pvec = np.zeros(fc.D)
    pvec[:4] = [lp_prior.rvs(), llam_prior.rvs(),
                rv_prior.rvs(), ebv_prior.rvs()]
    pvec[4:] = np.asarray([sedw(*x) for x in fc.xstar])
    pvec[4:] += np.random.normal(0, 0.001, size=fc.nph * fc.nl)
    pvecs.append(pvec)

pvecs[0][0] = lp
pvecs[0][1] = llam
pvecs[0][2] = rv
pvecs[0][3] = ebv

surf2 = fc._create_model(bolomc.ParamVec(pvecs[0], fc.nph, fc.nl)).source._passed_flux / fc.hsiao._passed_flux

surf2[np.isnan(surf2)] = 0.

source2 = sncosmo.TimeSeriesSource(fc.hsiao._phase,
                                   fc.hsiao._wave,
                                   surf2)

pickle.dump(source2, open("init.pkl",'wb'))

bolomc.main(fc, 'sine.h52',
            1000, 1000,
            nwalkers=nwalkers,
            nthreads=1,
            fc_fname='sinefc.pkl2',
            logfile='sine.log2',
            pvecs=pvecs)
