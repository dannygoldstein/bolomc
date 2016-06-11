#!/usr/bin/env python

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Make bolometric light curves for csp sne ia."

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
import glob
import h5py
from bolomc import CSPFitContext, ParamVec, bolo
import pickle

sns.set_style('ticks')    

targets = glob.glob('../run/*h5_lores')

fig, ax = plt.subplots()

for target in targets:
    print target
    stack = bolo.LCStack.from_hdf5(target)
    
    stack.plot(ax=ax)
fig.savefig('allbolo.pdf')
