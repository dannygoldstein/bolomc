#!/usr/bin/env python

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Test problem with a uniform warping surface."

import matplotlib
matplotlib.use("Agg")
import sncosmo
import problem
import numpy as np

def sedw(p, l):
    return 1 + 0.5 * np.sin((np.pi * p / 20) + (np.pi * l / 1500)) 

# True parameters.
rv = 3.1 # Host.
ebv = 0.1 # Host.
lp = 20 # Days. TODO:: Verify this. 
llam = 1500 # Angstrom. TODO:: Verify this.
z = 0.02 
mwebv = 0.02 

test = problem.Problem(sedw, rv, ebv, lp, llam, z, mwebv)
data = test.data(100/8, exclude_bands=['csphd','cspjd','cspyd',
                                       'cspv3014','cspv3009'])
sources = [problem.hsiao, test.source]
fig = sncosmo.plot_lc(data=data)

sncosmo.animate_source(sources=sources, fname='sine.mp4')
