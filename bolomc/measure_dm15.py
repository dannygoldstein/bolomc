#!/usr/bin/env python

__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Meausre dm15 and peak L for CSP SNe Ia."

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
import glob
import h5py
from bolomc import CSPFitContext, ParamVec
import pickle

sns.set_style('ticks')

targets = glob.glob('../run/*h5_lores')

lowdm15 = []
meddm15 = []
uppdm15 = []

lowL = []
medL = []
uppL = []

gals = []

LOWPCT = 50 - 68 / 2.
HIPCT = 50 + 68 / 2.

def check_good(name):
    with open('../run/goodfits.dat','r') as f:
        for line in f:
            if name in line:
                if 'good' in line:
                    return True
    return False

def get_galtype(name):
    with open('../data/gal.dat','r') as f:
        for line in f:
            if name in line:
                gtype = line.split("\" ")[-1]
                if gtype.startswith("E"):
                    return 'E'
                if gtype.startswith("S"):
                    return "S"
                else:
                    return None

for target in targets:

    name = target.split('.h5')[0].split('/')[-1]
    
    print name
    good = check_good(name)

    if not good:
        continue

    galtype = get_galtype(name)

    this_dm15 = []
    this_L = []

    try:
        f = h5py.File(target)
    except IOError:
        continue
    fc = pickle.load(open(target.replace('h5', 'fc.pkl'), 'rb'))
    if f['current_stage'][()]:
        try:
            p = np.vstack(f['samples']['params'][[0, 49]])
        except:
            i = f['samples']['last_index_filled'][()]
            p = f['samples']['params'][i]
    else:
        i = f['burn']['last_index_filled'][()]
        p = f['burn']['params'][i]
    for param in p:
        pvec = ParamVec(param, fc.nph, fc.nl)
        try:
            d = fc.dm15(pvec)[0]
            l = fc.Lpeak(pvec)[0]
        except:
            continue
        else:
            this_dm15.append(d)
            this_L.append(l)

    lowdm15.append(np.percentile(this_dm15, LOWPCT))
    meddm15.append(np.median(this_dm15))
    uppdm15.append(np.percentile(this_dm15, HIPCT))
    
    lowL.append(np.percentile(this_L, LOWPCT))
    medL.append(np.median(this_L))
    uppL.append(np.percentile(this_L, HIPCT))
    
    gals.append(galtype)
    
def color_gal(gal):
    if gal == 'E':
        return 'r'
    elif gal == 'S':
        return 'b'
    else:
        return 'k'

# plotting

lowdm15 = np.asarray(lowdm15)
meddm15 = np.asarray(meddm15)
uppdm15 = np.asarray(uppdm15)

lowL = np.asarray(lowL)
medL = np.asarray(medL)
uppL = np.asarray(uppL)

fig, ax = plt.subplots()

gals = map(color_gal, gals)

for i in range(len(gals)):
    ax.errorbar(meddm15[i], medL[i], xerr=[[meddm15[i] - lowdm15[i]], [uppdm15[i] - meddm15[i]]],
                yerr=[[medL[i] - lowL[i]], [uppL[i] - medL[i]]], fmt='.', color=gals[i])

ax.set_ylim(1e42, 5e43)
ax.set_yscale('log')
ax.set_ylabel('log peak luminosity (erg / s)')
ax.set_xlabel('dm15 (mag)')
sns.despine(ax=ax)
fig.savefig('dm15.pdf')
