import h5py
import matplotlib
matplotlib.use("Agg")
import pickle
from bolomc import plotting
from bolomc import CSPFitContext
import numpy as np
import glob

import seaborn as sns


sns.set_style('ticks')    

targets = glob.glob('../run/*h5_lores')

for target in targets:

    target = target.split('.h5')[0]

    fc = pickle.load(open(target + '.fc.pkl_lores','rb'))

    p = fc.hsiao._phase
    l = fc.hsiao._wave

    f = h5py.File(target + '.h5_lores')

    W = f['burn']['params'][0, :, 2:].reshape(-1, fc.nph, fc.nl)
    W = np.asarray([fc._regrid_hsiao(warp) for warp in W])
    
    wstd = W.std(0)
    wmed = np.median(W, 0)
    

    fig = plotting.plot_wsurf(p, l, wmed, lc=fc.lc)
    
    ax = fig.gca()
    
    ax.plot(*fc.xstar.T[::-1], color='r', marker='x',
             linestyle='none')
    fig.savefig(target+'.surf.pdf')
    
    fig = plotting.plot_wsurf(p, l, wstd, lc=fc.lc)
    
    ax = fig.gca()
    
    ax.plot(*fc.xstar.T[::-1], color='r', marker='x',
             linestyle='none')
    fig.savefig(target+'.std.pdf')
    
    

