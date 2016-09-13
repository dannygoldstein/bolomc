from book import samples, plot_photmag

lc,_,models = samples.models('run/SN2005ag.out')
fig = plot_photmag(models, lc)
fig.savefig('agtest.pdf')
