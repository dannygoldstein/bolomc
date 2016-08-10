import matplotlib
matplotlib.use("Agg")
import glob
import samples
import numpy as np
import sncosmo
import pickle
from bolomc import bolo
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.cosmology import Planck13

files = glob.glob('run/*.out')
results = map(samples.models, files)
csp = sncosmo.get_magsystem('csp')
fitres = np.genfromtxt('run/fitres.dat', names=True, dtype=None)

# SNe that had fits that did not fail
good = fitres[fitres['status'] == 'OK']['name']

# keep only successful fits
results = filter(lambda tup: tup[0].meta['name'] in good, 
                 results)

def wlr(x, y, cov, xlim=None, ylim=None, band='B'):
    """Plot the width-luminosity relation."""

    import seaborn as sns
    sns.set_style('ticks')
    fig, ax = plt.subplots()

    xe = [np.sqrt(e[0,0]) for e in cov]
    ye = [np.sqrt(e[1,1]) for e in cov]

    ax.errorbar(x, y, xerr=xe, yerr=ye, capsize=0, color='k', 
                linestyle='None')

    ax.set_xlabel(r'$\Delta m_{15}(%s)$' % band)
    ax.set_ylabel(r'$M_{\mathrm{abs}}(%s)$' % band)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.invert_yaxis()

    z = np.asarray(zip(x, y))
    sigma = cov 

    def obj_func((m, b, V)):
        theta = np.arctan(m)
        v = np.asarray([-np.sin(theta), np.cos(theta)])
        deltasq = (v.dot(z.T) - b * np.cos(theta))**2
        sigmasq = np.asarray([v.dot(sig).dot(v) for sig in sigma])
        return np.sum(deltasq / (sigmasq + V) + np.log(sigmasq + V))

    res = minimize(obj_func, (.7, -20., 0.15**2))
    x = np.linspace(.75, 1.75)
    y = res.x[0] * x + res.x[1]

    # show intrinsic scatter
    off = np.sqrt(res.x[2]) / np.sin(np.arctan(res.x[0]))
    ax.fill_between(x, y + off, y - off, color='r', alpha=0.2)
    ax.plot(x, y, 'r')
    sns.despine(ax=ax)
    ax.set_title(r'$m=%.2f, b=%.2f, \sigma=%.2f$' % 
                 (res.x[0], res.x[1], np.sqrt(res.x[2])))
 
    return fig

"""
# broadband book
with PdfPages('phot.pdf') as pdf:
    for (lc, config, models) in results:
        fig = sncosmo.plot_lc(model=models, data=lc, ci=(2.5, 50., 97.5),
                              figtext=lc.meta['name'])
        pdf.savefig(fig)
"""
"""
# bolometric book        
with PdfPages('bolo.pdf') as pdf:
    from bolomc import bolo
    for (lc, config, models) in results:
        stack = bolo.LCStack.from_models(models)
        ax = stack.plot()
        ax.set_title(lc.meta['name'])
        ax.set_ylim(0, 2.5e43)
        pdf.savefig(ax.figure)

# wlr plot
dm15 = []; M = []; cov = []
for (lc, config, models) in results:
    try: 
        tdm15 = np.squeeze(map(bolo.dm15, models))
        tL = np.squeeze(map(bolo.Lpeak, models))
        # arbitrary zero point
        tM = np.squeeze([-2.5 * np.log10(L) + 87.3 for L in tL])

    except ValueError:
        continue
    dm15.append(np.mean(tdm15))
    M.append(np.mean(tM))
    cov.append(np.cov(zip(tdm15, tM), rowvar=False))
fig = wlr(dm15, M, cov, band='bol')
fig.savefig('wlr.pdf')

"""

dm15 = []; M = []; cov = []
for (lc, config, models) in results:
    if lc.meta['name'] not in good:
        continue
    tdm15 = []
    tM = []
    for model in models:
        peakmag = model.source_peakabsmag('cspb', csp, cosmo=Planck13)
        peakphase = model.source.peakphase('cspb')
        mag0 = model.source.bandmag('cspb', csp, peakphase)
        mag15 = model.source.bandmag('cspb', csp, peakphase+15)
        tdm15.append(mag15 - mag0)
        tM.append(peakmag)
    if np.mean(tdm15) < 2.:
        dm15.append(np.mean(tdm15))
        M.append(np.mean(tM))
        cov.append(np.cov(zip(tdm15, tM), rowvar=False))

fig = wlr(dm15, M, cov, xlim=(.75, 1.75), 
          ylim=(-19.7, -18.4))
fig.savefig('bbwlr.pdf')
