import matplotlib
matplotlib.use("Agg")
import sncosmo
import glob
import samples
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

files = glob.glob('run/*.out')
results = map(samples.models, files)

# bolometric book

with PdfPages('phot.pdf') as pdf:
    for (lc, config, models) in results:
        fig = sncosmo.plot_lc(model=models, data=lc, ci=(2.5, 50., 97.5),
                              figtext=lc.meta['name'])
        pdf.savefig(fig)
        
with PdfPages('bolo.pdf') as pdf:
    from bolomc import bolo
    for (lc, config, models) in results:
        stack = bolo.LCStack.from_models(models, name=lc.meta['name'])
        ax = stack.plot()
        pdf.savefig(ax.figure)

fig, ax = plt.subplots()
dm15 = L = []
dm15e = Le = []
for (lc, config, models) in results:
    tdm15 = map(bolo.dm15, models)
    tL = map(bolo.Lpeak, models)
    dm15.append(np.mean(tdm15))
    L.append(np.mean(tL))
    dm15e.append(np.std(tdm15))
    Le.append(np.std(tL))
ax.errorbar(dm15, L, xerr=dm15e, yerr=Le, fmt='.', capsize=0)
fig.savefig('wlr.pdf')    


