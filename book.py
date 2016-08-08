import matplotlib
matplotlib.use("Agg")
import glob
import samples
import numpy as np
import sncosmo
from bolomc import bolo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

files = glob.glob('run/*.out')
results = map(samples.models, files)
"""
# broadband book
with PdfPages('phot.pdf') as pdf:
    for (lc, config, models) in results:
        fig = sncosmo.plot_lc(model=models, data=lc, ci=(2.5, 50., 97.5),
                              figtext=lc.meta['name'])
        pdf.savefig(fig)

# bolometric book        
with PdfPages('bolo.pdf') as pdf:
    from bolomc import bolo
    for (lc, config, models) in results:
        stack = bolo.LCStack.from_models(models)
        ax = stack.plot()
        ax.set_title(lc.meta['name'])
        pdf.savefig(ax.figure)
"""
# wlr plot
fig, ax = plt.subplots()
dm15 = []; L = []
dm15e = []; Le = []
gal = np.genfromtxt('data/gal.dat', delimiter='"')
gals = []
for (lc, config, models) in results:
    try: 
        tdm15 = map(bolo.dm15, models)
        tL = map(bolo.Lpeak, models)
    except ValueError:
        continue
    dm15.append(np.mean(tdm15))
    L.append(np.mean(tL))
    dm15e.append(np.std(tdm15))
    Le.append(np.std(tL))
    try:
        gtype = gal[gal['name'] == lc.meta['name']]['host_type'][0]
    except:
        gtype = 'unknown'
    else: 
        gtype = 'S' if 'S' in gtype else 'E'
    gals.append(gtype)
gals = np.asarray(gals)
for gtype in set(gals):
    d = np.asarray(dm15)[gals==gtype]
    l = np.asarray(L)[gals==gtype]
    de = np.asarray(dm15e)[gals==gtype]
    le = np.asarray(Le)[gals==gtype]
    if 'S' == gtype:
        color = 'blue'
        label = 'spiral'
    elif 'E' == gtype:
        color = 'red'
        label = 'elliptical'
    else:
        color = 'k'
        label = 'unknown'

    ax.errorbar(d, l, xerr=de, yerr=le, fmt='.', capsize=0, color=color,
                label=label)

ax.set_xlim(0.5, 1.5)
ax.set_ylim(0.2e43, 2.2e43)
ax.legend()
import seaborn as sns
sns.set_style('ticks')
sns.despine(ax=ax)
fig.savefig('wlr.pdf')    
